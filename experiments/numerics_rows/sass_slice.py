"""Compare quotient dependency graphs from the loaded binaries. Refuse unknown inputs."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from experiments.numerics_rows.common import sha256, write_json
from experiments.numerics_rows.reference import f32_value, rn_bits
from experiments.t4_code.sass_dump import parse_sass

REG = re.compile(r"\b(?:UR|UP|R|P)(?:\d+|Z|T)\b")
CONST = re.compile(r"c\[(0x[0-9a-f]+)\]\[(0x[0-9a-f]+)\]", re.I)
ADDR = re.compile(r"/\*([0-9a-f]{4,})\*/", re.I)
MODE0 = re.compile(r"routed_fused_(?:fp4_)?kernelI(?:Lb[01]E)?Li0E")
ALU = re.compile(r"^(?:U?(?:IADD3|IADD|IMAD|ISETP|MOV|LOP3|SHF|LEA|FLO)|FSETP|DSETP|PSETP|PLOP3|FCHK|FADD|FMUL|FFMA|FMNMX|MUFU|HFMA2|F2F|F2FP|F2I|I2F|SEL|PRMT|BMSK)(?:\.|$)")


class Unmatched(ValueError):
    pass


def constants(text):
    """Read little-endian words from cuobjdump's ELF constant sections."""
    result, section, offset, relocation = {}, None, 0, None
    for line in text.splitlines():
        header = re.fullmatch(r"\s*\.nv\.constant(\d+)(?:\.([^\s:]+))?\s*:?\s*", line)
        if header:
            section = (int(header[1]), header[2] or "")
            if section in result:
                raise Unmatched(f"duplicate constant section: {section}")
            result[section], offset, relocation = {}, 0, None
            continue
        if line.lstrip().startswith("."):
            section = None  # Relocations, symbol tables and other sections are not bytes.
            reloc = re.fullmatch(r"\s*\.section \.rela\.nv\.constant(\d+)(?:\.(\S+))?\s+RELA\s*", line)
            relocation = (int(reloc[1]), reloc[2] or "") if reloc else None
        if relocation is not None and "R_CUDA_" in line:
            result[relocation] = {}  # Relocated addresses are not resolved constant bytes.
        words = re.fullmatch(r"\s*((?:0x[0-9a-f]{8}\s*)+)\s*", line, re.I)
        if section is not None and words:
            # cuobjdump -elf prints raw words, with no address in the first column.
            data = b"".join(struct.pack("<I", int(w, 16)) for w in words[1].split())
            for i, value in enumerate(data):
                result[section][offset + i] = value
            offset += len(data)
    return result


def cubins(text):
    """Keep duplicate symbols and banks in separate fatbinary ELF records."""
    chunks = re.split(r"(?m)^Fatbin elf code:\s*$", text)
    if len(chunks) == 1:
        return {("fixture", "", 0): text}
    result, occurrences = {}, {}
    for chunk in chunks[1:]:
        arch = re.search(r"(?m)^arch\s*=\s*(\S+)", chunk)
        identifier = re.search(r"(?m)^identifier\s*=\s*(.*?)\s*$", chunk)
        if not arch or not identifier:
            raise Unmatched("fatbinary ELF record lacks its architecture or identifier")
        key = arch[1], identifier[1]
        ordinal = occurrences.get(key, 0)
        occurrences[key] = ordinal + 1
        result[(*key, ordinal)] = chunk
    return result


@dataclass(frozen=True)
class Instruction:
    address: int
    predicate: str
    opcode: str
    operands: tuple

    @property
    def destination_count(self):
        if self.opcode.startswith(("ST", "BRA", "BRX", "JMP", "JMX", "CALL", "JCAL", "RET", "EXIT", "BSSY", "BSYNC", "BAR", "WARPSYNC", "MEMBAR", "ERRBAR", "DEPBAR", "NOP", "NANOSLEEP", "CCTL")):
            return 0
        if self.opcode.startswith(("ISETP", "FSETP", "DSETP", "PSETP", "UISETP", "PLOP3")):
            return 2
        if self.opcode.startswith(("IADD3", "UIADD3")):
            return 1 + sum(bool(re.fullmatch(r"U?P(?:\d+|T)", x)) for x in self.operands[1:3])
        if self.opcode.startswith("LOP3") and re.fullmatch(r"P(?:\d+|T)", self.operands[0]):
            return 2
        if self.opcode.startswith(("LEA", "ULEA", "SHFL")) and len(self.operands) > 1 and re.fullmatch(r"U?P(?:\d+|T)", self.operands[1]):
            return 2
        return 1

    @property
    def destinations(self):
        return tuple(r for operand in self.operands[:self.destination_count] for r in REG.findall(operand))

    @property
    def text(self):
        return f"/*{self.address:04x}*/ {self.predicate + ' ' if self.predicate else ''}{self.opcode} {', '.join(self.operands)};"


def instructions(text):
    parsed = parse_sass(text)
    addresses, name = {}, None
    for line in text.splitlines():
        function = re.match(r"\s*Function\s*:\s*(\S+)", line)
        if function:
            name = function[1]
            if name in addresses:
                raise Unmatched(f"duplicate SASS function in one cubin: {name}")
            addresses[name] = []
        address = ADDR.search(line)
        if name and address:
            addresses[name].append(int(address[1], 16))
    result = {}
    for name, sequence in parsed.items():
        if len(addresses[name]) != len(sequence):
            raise Unmatched(f"address parse differs: {name}")
        result[name] = []
        for address, line in zip(addresses[name], sequence):
            match = re.fullmatch(r"(?:(@!?U?P(?:\d+|T))\s+)?(\S+)(?:\s+(.*))?", line)
            if not match:
                raise Unmatched(f"instruction parse failed: {line}")
            result[name].append(Instruction(address, match[1] or "", match[2],
                                            tuple(x.strip() for x in (match[3] or "").split(","))))
    return result


def rounded_float(operation, values, *, half=False, ftz=False):
    """Evaluate one fused or single IEEE RN operation with exact rational arithmetic."""
    precision, emin, emax = (11, -14, 15) if half else (24, -126, 127)
    if ftz:
        values = [math.copysign(0.0, x) if abs(x) < 2.0**emin else x for x in values]
    if any(math.isnan(x) for x in values):
        raise ValueError("NaN payload is not resolved")
    if not all(math.isfinite(x) for x in values):
        value = values[0] + values[1] if operation == "FADD" else values[0] * values[1]
        if operation == "FFMA":
            value += values[2]
        if math.isnan(value):
            raise ValueError("NaN payload is not resolved")
    else:
        a, b, *rest = values
        value = Fraction(a) + Fraction(b) if operation == "FADD" else Fraction(a) * Fraction(b)
        if rest:
            value += Fraction(rest[0])
        if not value:
            product_sign = math.copysign(1.0, a) * math.copysign(1.0, b)
            negative = (product_sign < 0 if operation == "FMUL" else
                        product_sign < 0 and math.copysign(1.0, rest[0]) < 0 if rest else
                        math.copysign(1.0, a) < 0 and math.copysign(1.0, b) < 0)
            value = -0.0 if negative else 0.0
        value = f32_value(rn_bits(value, precision, emin, emax))
    if ftz and abs(value) < 2.0**emin:
        value = math.copysign(0.0, value)
    return int.from_bytes(struct.pack("<e" if half else "<f", value), "little")


def immediate32(ins, resolve=lambda text: None):
    """Fold only immediate, resolved-bank and zero-register inputs to exact bits."""
    operands = ins.operands[ins.destination_count:]
    opcode = ins.opcode
    if opcode in ("LOP3.LUT", "ULOP3.LUT") and operands[-1:] == ("!PT",):
        operands = operands[:-1]
    if any(REG.search(x) and not re.fullmatch(r"-?(?:RZ|URZ)(?:\.reuse)?", x) for x in operands):
        return None
    def integer(text):
        if re.fullmatch(r"-?(?:RZ|URZ)(?:\.reuse)?", text):
            return 0
        if CONST.fullmatch(text):
            value = resolve(text)
            if value is None:
                raise ValueError("unresolved constant")
            return value
        return int(text, 0) & 0xFFFFFFFF
    def floating(text, *, half=False):
        if re.fullmatch(r"-?(?:RZ|URZ)(?:\.reuse)?", text):
            return -0.0 if text.startswith("-") else 0.0
        if re.fullmatch(r"-0x[0-9a-f]+", text, re.I):
            return f32_value(int(text[1:], 0) ^ 0x80000000)
        if CONST.fullmatch(text) or re.fullmatch(r"0x[0-9a-f]+", text, re.I):
            return f32_value(integer(text))
        if text.upper() in ("INF", "+INF", "-INF"):
            return float(text)
        value = Fraction(text)
        if not value and text.startswith("-"):
            value = -0.0
        return f32_value(rn_bits(value, 11, -14, 15) if half else rn_bits(value))
    try:
        if opcode in ("MOV", "UMOV") and len(operands) == 1:
            return integer(operands[0])
        if opcode in ("HFMA2", "HFMA2.RN") and len(operands) in (3, 4):
            a, b = map(integer, operands[:2])
            a = 0x80008000 if re.fullmatch(r"-(?:RZ|URZ)(?:\.reuse)?", operands[0]) else a
            b = 0x80008000 if re.fullmatch(r"-(?:RZ|URZ)(?:\.reuse)?", operands[1]) else b
            additions = ([floating(operands[3], half=True), floating(operands[2], half=True)]
                         if len(operands) == 4 else struct.unpack("<ee", struct.pack("<I", integer(operands[2]))))
            halves = [rounded_float("FFMA", [struct.unpack("<e", struct.pack("<H", (a >> shift) & 65535))[0],
                       struct.unpack("<e", struct.pack("<H", (b >> shift) & 65535))[0], additions[i]], half=True)
                      for i, shift in enumerate((0, 16))]
            return halves[0] | (halves[1] << 16)
        operation, *modifiers = opcode.split(".")
        if operation in ("FFMA", "FADD", "FMUL") and set(modifiers) <= {"RN", "FTZ"}:
            if len(operands) == (3 if operation == "FFMA" else 2):
                return rounded_float(operation, list(map(floating, operands)), ftz="FTZ" in modifiers)
        if opcode in ("IMAD", "IMAD.U32", "IMAD.MOV.U32", "IMAD.SHL.U32", "IADD3", "UIADD3") and len(operands) == 3:
            a, b, c = map(integer, operands)
            return ((a + b + c) if operation in ("IADD3", "UIADD3") else (a * b + c)) & 0xFFFFFFFF
        if opcode in ("LOP3.LUT", "ULOP3.LUT") and len(operands) == 4:
            a, b, c, lut = map(integer, operands)
            return sum(((lut >> ((((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | ((c >> i) & 1))) & 1) << i for i in range(32))
        if opcode == "SHF.L.U32" and len(operands) == 3 and operands[2] in ("RZ", "URZ"):
            a, shift, _ = map(integer, operands)
            return (a << min(shift, 32)) & 0xFFFFFFFF
    except (ValueError, OverflowError, struct.error):
        pass
    return None


def left_shift(ins):
    """Recognize only the two shift forms permitted by the pass rule."""
    if ins.destination_count != 1 or len(ins.operands) != 4 or ins.operands[3] != "RZ":
        return None
    try:
        value = int(ins.operands[2], 0)
    except ValueError:
        return None
    if ins.opcode == "IMAD.SHL.U32" and 0 < value <= 2**31 and value & (value - 1) == 0:
        return ins.operands[1], value.bit_length() - 1
    if ins.opcode == "SHF.L.U32" and 0 <= value < 32:
        return ins.operands[1], value
    return None


class Slice:
    """Symbolically execute the quotient, including its IEEE division slow path.

    Branch arms rejoin at the requested value. Only dependencies of that value
    enter its graph. In-function CALL/RET use an explicit return stack.
    """

    def __init__(self, functions, const, name, gf, gf_after):
        self.functions, self.const, self.name = functions, const, name
        self.gf, self.gf_after = gf, gf_after
        self.resolved, self.nodes, self.initial, self.facts, self.simplified = {}, {}, {}, {}, {}
        self.origins = {}
        self.addresses = {name: {ins.address: i for i, ins in enumerate(functions[name])}}
        self.definitions = {}
        for i, ins in enumerate(functions[name]):
            for reg in ins.destinations:
                self.definitions.setdefault((name, reg), []).append(i)

    def node(self, kind, *args):
        value = (kind, *args)
        key = "$" + hashlib.sha256(json.dumps(value).encode()).hexdigest()
        self.nodes[key] = value
        return key

    def constant_word(self, operand, name, width=4):
        match = CONST.fullmatch(operand)
        if not match or int(match[1], 16) == 0:
            return None
        bank, offset = (int(x, 16) for x in match.groups())
        section = self.const.get((bank, name), self.const.get((bank, ""), {}))
        try:
            data = bytes(section[offset + i] for i in range(width))
        except KeyError:
            return None
        self.resolved[f"{name}:{operand}"] = data.hex()
        return int.from_bytes(data, "little")

    def operand(self, operand, name, index, state):
        unresolved = []
        def constant(match):
            bank, offset = (int(x, 16) for x in match.groups())
            if bank == 0:
                unresolved.append("c[0x0] kernel parameter")
                return ""
            width = 8 if self.functions[name][index].opcode.startswith("D") else 4
            word = self.constant_word(match[0], name, width)
            if word is None:
                unresolved.append(f"constant {match[0]} in {name}")
                return ""
            return "constant:" + word.to_bytes(width, "little").hex()
        operand = CONST.sub(constant, operand)
        if unresolved:
            return self.node("unresolved", unresolved[0])
        if "[" in operand or "]" in operand:
            return self.node("unresolved", f"indirect address: {operand}")
        if not REG.search(operand) and not re.fullmatch(r"[+-]?(?:constant:[0-9a-f]+|0x[0-9a-f]+|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|INF|QNAN)", operand, re.I):
            return self.node("unresolved", f"operand: {operand}")
        pieces, end = [], 0
        for match in REG.finditer(operand):
            pieces.extend((operand[end:match.start()], self.register(match[0], state)))
            end = match.end()
        pieces.append(operand[end:])
        return self.node("operand", tuple(pieces))

    def register(self, reg, state):
        if reg in ("RZ", "URZ", "PT", "UPT"):
            return self.node("fixed", reg)
        if reg in state:
            return self.simplify(state[reg])
        # Values defined before the clamp may supply constants, but no other input.
        return self.simplify(self.before(reg, self.gf_after + 1))

    def simplify(self, value):
        key = value, frozenset(self.facts.items())
        if key in self.simplified:
            return self.simplified[key]
        if value in self.facts:
            return self.node("boolean", self.facts[value])
        kind, *args = self.nodes[value]
        def rewrite(item):
            if isinstance(item, (list, tuple)):
                return tuple(rewrite(x) for x in item)
            return self.simplify(item) if isinstance(item, str) and item in self.nodes else item
        args = [rewrite(arg) for arg in args]
        if kind == "not" and self.nodes[args[0]][0] == "boolean":
            result = self.node("boolean", not self.nodes[args[0]][1])
        elif kind == "select" and self.nodes[args[0]][0] == "boolean":
            result = args[1] if self.nodes[args[0]][1] else args[2]
        elif kind == "select":
            condition, *arms = args
            for i, arm in enumerate(arms):
                while self.nodes[arm][0] == "select":
                    _, inner, yes, no = self.nodes[arm]
                    if inner == condition:
                        arm = yes if i == 0 else no
                    elif self.nodes[inner] == ("not", condition) or self.nodes[condition] == ("not", inner):
                        arm = no if i == 0 else yes
                    else:
                        break
                arms[i] = arm
            result = arms[0] if arms[0] == arms[1] else self.node(kind, condition, *arms)
        else:
            result = self.node(kind, *args)
        self.simplified[key] = result
        if value in self.origins:
            self.origins.setdefault(result, {}).update(self.origins[value])
        return result

    def predicate(self, text, state):
        reg = text.removeprefix("!")
        if not REG.fullmatch(reg):
            raise Unmatched(f"unresolved predicate: {text}")
        value = self.register(reg, state)
        return self.simplify(self.node("not", value)) if text.startswith("!") else value

    def before(self, reg, before):
        if reg == self.gf and before == self.gf_after + 1:
            return self.node("gf")
        key = reg, before
        if key in self.initial:
            return self.initial[key]
        positions = self.definitions.get((self.name, reg), [])
        pos = bisect_left(positions, before) - 1
        if pos < 0:
            return self.node("unresolved", f"undefined operand {reg}")
        index = positions[pos]
        if reg == self.gf and index == self.gf_after:
            return self.node("gf")
        ins = self.functions[self.name][index]
        if ins.opcode.startswith(("LD", "ULD", "S2R", "S2UR", "CS2R")) and not ins.opcode.startswith(("LDC", "ULDC")):
            return self.node("unresolved", f"load: {ins.opcode}")
        if any(op.opcode.startswith(("BRA", "BRX", "JMP", "CALL")) for op in self.functions[self.name][index + 1:before]):
            return self.node("unresolved", "definition before the clamp crosses control flow")
        state = {r: self.before(r, index) for x in ins.operands[ins.destination_count:] for r in REG.findall(x)
                 if r not in ("RZ", "URZ", "PT", "UPT")}
        value = self.operation(ins, self.name, index, state, ins.destinations.index(reg))
        if ins.predicate:
            value = self.node("select", self.predicate(ins.predicate[1:], state), value, self.before(reg, index))
        self.initial[key] = value
        return value

    def operation(self, ins, name, index, state, slot):
        value = self.operation_value(ins, name, index, state, slot)
        self.origins.setdefault(value, {})[(name, ins.address)] = ins
        return value

    def operation_value(self, ins, name, index, state, slot):
        constant = (immediate32(ins, lambda x: self.constant_word(x, name))
                    if re.fullmatch(r"U?R\d+", ins.destinations[slot]) else None)
        if constant is not None:
            return self.node("constant", constant)
        if ins.opcode == "MOV" and len(ins.operands) == 2 and re.fullmatch(r"(?:UR|R)\d+(?:\.reuse)?", ins.operands[1]):
            return self.register(ins.operands[1].removesuffix(".reuse"), state)
        shift = left_shift(ins)
        if shift is not None:
            return self.node("shift-left", self.operand(shift[0], name, index, state), shift[1])
        if ins.opcode.startswith(("LD", "ULD", "S2R", "S2UR", "CS2R")):
            if not ins.opcode.startswith(("LDC", "ULDC")) or not any(CONST.search(x) for x in ins.operands[ins.destination_count:]):
                return self.node("unresolved", f"load: {ins.opcode}")
        elif not ALU.match(ins.opcode):
            return self.node("unresolved", f"destination semantics: {ins.opcode}")
        inputs = tuple(self.operand(x, name, index, state) for x in ins.operands[ins.destination_count:])
        return self.node("op", ins.opcode, slot, inputs)

    def target(self, name, operand):
        try:
            if name not in self.addresses:
                self.addresses[name] = {ins.address: i for i, ins in enumerate(self.functions[name])}
            return name, self.addresses[name][int(operand, 16)]
        except (ValueError, KeyError):
            if operand in self.functions:
                return operand, 0
            raise Unmatched(f"unresolved control target: {operand}")

    def word(self, value):
        """Resolve the explicit MOV chain used for a CALL's 64-bit return PC."""
        node = self.nodes[value]
        if node[0] == "constant":
            return node[1]
        if node == ("fixed", "RZ"):
            return 0
        if node[0] == "operand":
            pieces = node[1]
            if len(pieces) == 1:
                return int(pieces[0], 0)
            if len(pieces) == 3 and pieces[0] in ("", "-") and not pieces[2]:
                return self.word(pieces[1]) * (-1 if pieces[0] else 1)
        if node[0] == "op" and node[1] in ("MOV", "UMOV"):
            return self.word(node[3][0])
        if node[0] == "op" and node[1] == "HFMA2" and all(self.word(x) == 0 for x in node[3]):
            return 0
        raise Unmatched("unresolved return address")

    def execute(self, name, pc, state, stack, visited, facts):
        seq = self.functions[name]
        visited = set(visited)
        while 0 <= pc < len(seq):
            self.facts = facts
            if not stack and name == self.name and pc == self.stop:
                return self.register(self.output, state)
            key = name, pc, tuple((n, i) for n, i, *_ in stack)
            if key in visited:
                raise Unmatched("cyclic quotient control flow")
            visited.add(key)
            ins = seq[pc]
            if ins.opcode.startswith(("BRA", "JMP", "BRX", "JMX")):
                target_name, target_pc = self.target(name, ins.operands[-1])
                if target_name != name:
                    raise Unmatched("branch leaves its function")
                if ins.predicate or len(ins.operands) > 1:
                    guard = ins.predicate.removeprefix("@") or ins.operands[0]
                    value = self.predicate(guard, state)
                    if self.nodes[value][0] == "boolean":
                        if not self.nodes[value][1]:
                            pc += 1
                            continue
                        if not ins.predicate or len(ins.operands) == 1:
                            pc = target_pc
                            continue
                    condition = (value,
                                 tuple(self.operand(x, name, pc, state) for x in ins.operands[:-1]))
                    guard_reg = guard.removeprefix("!")
                    reg_value = state[guard_reg] if guard_reg in state else self.before(guard_reg, self.gf_after + 1)
                    taken_facts = {**facts, reg_value: not guard.startswith("!")}
                    other_facts = {**facts, reg_value: guard.startswith("!")} if not ins.predicate or len(ins.operands) == 1 else facts
                    taken = self.execute(name, target_pc, dict(state), stack, visited, taken_facts)
                    other = self.execute(name, pc + 1, dict(state), stack, visited, other_facts)
                    return taken if taken == other else self.node("branch", ins.opcode, condition, taken, other)
                return self.node("jump", ins.opcode, self.execute(name, target_pc, dict(state), stack, visited, facts))
            if ins.opcode.startswith(("CALL", "JCAL")):
                if ins.predicate:
                    raise Unmatched("predicated CALL needs an explicit control split")
                target_name, target_pc = self.target(name, ins.operands[-1])
                ret = seq[pc - 1]
                if ret.opcode != "MOV" or int(ret.operands[1], 16) != seq[pc + 1].address:
                    raise Unmatched("CALL return address is not an explicit MOV")
                stack = (*stack, (name, pc + 1, dict(state), ins.opcode, ret.operands[0]))
                name, pc, seq = target_name, target_pc, self.functions[target_name]
                continue
            if ins.opcode.startswith("RET"):
                if not stack or ins.predicate:
                    raise Unmatched("unresolved or predicated return")
                caller, next_pc, previous, call_op, return_reg = stack[-1]
                address = re.fullmatch(r"(R\d+) 0x0", ins.operands[0])
                if address is None:
                    raise Unmatched("unresolved RET operands")
                low, high = address[1], "R" + str(int(address[1][1:]) + 1)
                if (self.word(self.register(low, state)) != self.functions[caller][next_pc].address
                        or self.word(self.register(high, state)) != 0):
                    raise Unmatched("callee does not return to the CALL continuation")
                for reg in set(state) - {return_reg}:
                    if state[reg] != previous.get(reg):
                        state[reg] = self.node("call", call_op, ins.opcode, state[reg])
                name, pc, seq, stack = caller, next_pc, self.functions[caller], stack[:-1]
                continue
            if ins.opcode.startswith("EXIT"):
                raise Unmatched("quotient path exits before its value")
            updates = {}
            for slot, reg in enumerate(ins.destinations):
                if reg in ("RZ", "URZ", "PT", "UPT"):
                    continue
                value = self.operation(ins, name, pc, state, slot)
                if ins.predicate:
                    value = self.node("select", self.predicate(ins.predicate[1:], state), value, self.register(reg, state))
                updates[reg] = value
            state.update(updates)
            pc += 1
        raise Unmatched("quotient path leaves the parsed function")

    def root(self, before, reg):
        self.stop, self.output = before, reg
        value = self.execute(self.name, self.gf_after + 1, {self.gf: self.node("gf")}, (), set(), {})
        pending, seen = [value], set()
        while pending:
            item = pending.pop()
            if isinstance(item, (list, tuple)):
                pending.extend(item)
            elif isinstance(item, str) and item in self.nodes and item not in seen:
                seen.add(item)
                node = self.nodes[item]
                if node[0] == "unresolved":
                    raise Unmatched(node[1])
                pending.extend(node[1:])
        self.used = seen
        return value


def reaches_store(sequence, index, reg):
    live = {reg}
    for ins in sequence[index + 1:]:
        sources = set(REG.findall(",".join(ins.operands[len(ins.destinations):])))
        if ins.opcode.startswith("STG") and live.intersection(REG.findall(",".join(ins.operands[1:]))):
            return True
        destinations = set(ins.destinations)
        if sources & live:
            live |= destinations
        else:
            live -= destinations
    return False


def quotient_regions(functions, name, *, probe=False):
    seq = functions[name]
    checks = [i for i, ins in enumerate(seq) if ins.opcode == "FCHK"]
    if not checks:
        boundaries = [(i, r) for i, ins in enumerate(seq) if ins.opcode.startswith("LDG" if probe else "FMNMX")
                      for r in ins.destinations]
        return [(boundary, gf, i + 1, reg) for i, ins in enumerate(seq)
                if ins.opcode.startswith(("FMUL", "FFMA")) for reg in ins.destinations
                if reaches_store(seq, i, reg) for boundary, gf in boundaries if boundary < i]
    addresses = {ins.address: i for i, ins in enumerate(seq)}
    result = []
    for check in checks:
        predicate, gf = seq[check].operands[:2]
        boundary = next((i for i in range(check - 1, -1, -1) if gf in seq[i].destinations), None)
        if boundary is None or not seq[boundary].opcode.startswith("LDG" if probe else "FMNMX"):
            continue
        branch = next((i for i in range(check + 1, len(seq)) if seq[i].opcode.startswith("BRA")
                       and predicate in REG.findall(seq[i].predicate)), None)
        if branch is None:
            raise Unmatched("FCHK has no resolved slow-path branch")
        join = addresses.get(int(seq[branch].operands[-1], 16))
        if join is None or join <= branch:
            raise Unmatched("division slow path has no forward join")
        calls = [i for i in range(branch + 1, join) if seq[i].opcode.startswith("CALL")]
        if len(calls) != 1:
            raise Unmatched("division slow-path join needs one CALL")
        final = next((ins for ins in reversed(seq[check + 1:branch]) if ins.opcode.startswith(("FFMA", "FMUL"))), None)
        if final is None:
            raise Unmatched("division fast path has no final multiply or refinement")
        reg, = final.destinations
        if not reaches_store(seq, join - 1, reg):
            raise Unmatched("quotient does not feed an act store")
        result.append((boundary, gf, join, reg))
    return result


def binary_records(sass, elf):
    assembly, sections = cubins(sass), cubins(elf)
    if assembly.keys() != sections.keys():
        raise Unmatched("SASS and ELF cubin identities differ")
    return [(key, instructions(text), constants(sections[key])) for key, text in assembly.items()]


def difference(left, right, left_nodes, right_nodes, left_origins=None, right_origins=None):
    pending = [(left, right, [], {}, {})]
    while pending:
        a, b, path, left_source, right_source = pending.pop()
        if a == b:
            continue
        if isinstance(a, str) and a in left_nodes:
            left_source = (left_origins or {}).get(a, left_source)
            a = left_nodes[a]
        if isinstance(b, str) and b in right_nodes:
            right_source = (right_origins or {}).get(b, right_source)
            b = right_nodes[b]
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
            pending.extend((a[i], b[i], path + [i], left_source, right_source) for i in reversed(range(len(a))))
        else:
            return {"path": path, "probe": a, "library": b,
                    "probe_instructions": [{"function": name, "text": ins.text} for (name, _), ins in left_source.items()],
                    "library_instructions": [{"function": name, "text": ins.text} for (name, _), ins in right_source.items()]}


def instruction_examples(result, probe, library):
    """Keep three full instruction pairs for each approved instruction selection."""
    examples = result["instruction_examples"]
    for key in sorted(probe.used & library.used):
        if all(len(v) >= 3 for v in examples.values()):
            break
        for (pname, _), p in probe.origins.get(key, {}).items():
            for (lname, _), lib in library.origins.get(key, {}).items():
                kind = ("constant_nodes" if {p.opcode, lib.opcode} == {"FFMA", "MOV"} and library.nodes[key][0] == "constant" else
                        "register_moves" if "MOV" in {p.opcode, lib.opcode} and p.opcode != lib.opcode else
                        "left_shifts" if {p.opcode, lib.opcode} == {"IMAD.SHL.U32", "SHF.L.U32"} else None)
                if kind and len(examples[kind]) < 3:
                    record = {"probe_function": pname, "probe_instruction": p.text,
                              "library_function": lname, "library_instruction": lib.text,
                              "canonical_node": library.nodes[key]}
                    if record not in examples[kind]:
                        examples[kind].append(record)


def prove(library_sass, library_elf, probe_sass, probe_elf):
    """Require every mode-0 quotient, including all slow paths, to match."""
    result = {"mode0_instantiations": 0, "quotients": 0, "quotients_examined": 0,
              "constant_operands": {}, "unresolved": [], "differences": [],
              "instruction_examples": {"constant_nodes": [], "left_shifts": [], "register_moves": []},
              "matched": False, "status": "UNSUPPORTED"}
    try:
        libraries = binary_records(library_sass, library_elf)
        probes = binary_records(probe_sass, probe_elf)
        result["mode0_instantiations"] = sum(bool(MODE0.search(n)) for _, functions, _ in libraries for n in functions)
        if not result["mode0_instantiations"]:
            raise Unmatched("no mode-0 instantiations")
        expected = {}
        for key, functions, const in probes:
            for name in functions:
                if "swiglu_probe" not in name:
                    continue
                candidates = []
                for boundary, gf, stop, reg in quotient_regions(functions, name, probe=True):
                    sliced = Slice(functions, const, name, gf, boundary)
                    graph = sliced.root(stop, reg)
                    opcodes = {sliced.nodes[n][1] for n in sliced.used if sliced.nodes[n][0] == "op"}
                    if "MUFU.EX2" in opcodes and "MUFU.RCP" in opcodes:
                        candidates.append((graph, sliced))
                if len(candidates) != 1:
                    raise Unmatched(f"{key}: probe quotient candidates: {len(candidates)}")
                if key[0] in expected and expected[key[0]][0] != candidates[0][0]:
                    raise Unmatched("different probe quotient graphs for the same architecture")
                expected[key[0]] = candidates[0]
        if not expected:
            raise Unmatched("no parsed probe quotient")
        for key, functions, const in libraries:
            for name, seq in functions.items():
                if not MODE0.search(name):
                    continue
                try:
                    if key[0] not in expected:
                        raise Unmatched(f"no probe for {key[0]}")
                    wanted, probe = expected[key[0]]
                    result["constant_operands"].update(probe.resolved)
                    regions = quotient_regions(functions, name)
                    examined = 0
                    boundaries = set()
                    for boundary, gf, stop, reg in regions:
                        if boundary in boundaries:
                            continue
                        sliced = Slice(functions, const, name, gf, boundary)
                        try:
                            graph = sliced.root(stop, reg)
                        except Unmatched:
                            if any(ins.opcode == "FCHK" for ins in seq):
                                raise
                            continue
                        opcodes = {sliced.nodes[n][1] for n in sliced.used if sliced.nodes[n][0] == "op"}
                        if "MUFU.EX2" not in opcodes or "MUFU.RCP" not in opcodes:
                            continue
                        examined += 1
                        result["quotients_examined"] += 1
                        boundaries.add(boundary)
                        result["constant_operands"].update(sliced.resolved)
                        instruction_examples(result, probe, sliced)
                        if graph == wanted:
                            result["quotients"] += 1
                        else:
                            result["differences"].append({"cubin": list(key), "function": name,
                                "gf_pc": seq[boundary].address, "join_pc": seq[stop].address,
                                "probe_graph_sha256": wanted[1:], "library_graph_sha256": graph[1:],
                                "first_difference": difference(wanted, graph, probe.nodes, sliced.nodes, probe.origins, sliced.origins)})
                    exponentials = sum(ins.opcode == "MUFU.EX2" for ins in seq)
                    if examined != exponentials or not examined:
                        raise Unmatched(f"{examined} quotient graphs for {exponentials} exponential regions")
                except (Unmatched, ValueError, RecursionError) as error:
                    result["unresolved"].append(f"{name}: {error}")
        result["matched"] = not result["unresolved"] and not result["differences"]
        result["status"] = "MATCHED" if result["matched"] else "UNSUPPORTED" if result["unresolved"] else "DIFFERENT"
    except (Unmatched, ValueError, RecursionError) as error:
        result["unresolved"].append(str(error))
    return result


def match_library(record):
    for label in ("library", "probe"):
        if sha256(record[label]["path"]) != record[label]["sha256"]:
            raise ValueError(f"changed {label} binary")
        for kind in ("sass", "elf"):
            if sha256(record[label][kind]["path"]) != record[label][kind]["sha256"]:
                raise ValueError(f"changed {label} {kind} evidence")
    result = prove(*(Path(record[label][kind]["path"]).read_text() for label, kind in
                     (("library", "sass"), ("library", "elf"), ("probe", "sass"), ("probe", "elf"))))
    return {"library_so_sha256": record["library"]["sha256"], "probe_sha256": record["probe"]["sha256"], **result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    records = {r["key"]: match_library(r) for path in args.build for r in json.loads(path.read_text())["libraries"]}
    write_json(args.out, records)
    raise SystemExit(0 if all(r["matched"] for r in records.values()) else 3)
