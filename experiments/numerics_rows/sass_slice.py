"""Compare quotient dependency graphs from the loaded binaries. Refuse unknown inputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import subprocess

from experiments.numerics_rows.common import sha256, write_json
from experiments.t4_code.sass_dump import parse_sass

REG = re.compile(r"\b(?:UR|UP|R|P)(?:\d+|Z|T)\b")
CONST = re.compile(r"c\[(0x[0-9a-f]+)\]\[(0x[0-9a-f]+)\]", re.I)
ADDR = re.compile(r"/\*([0-9a-f]{4,})\*/", re.I)
WRITES = re.compile(r"^(?:U?I|U?F|U?D|U?H|U?P|U?L|U?S|MUFU|MOV|PRMT|BMSK|SHF|SEL|V|LEA|LOP|R2|P2)")


class Unmatched(ValueError):
    pass


def constants(text):
    """Read little-endian words from cuobjdump's ELF constant sections."""
    result, section = {}, None
    for line in text.splitlines():
        header = re.search(r"(?:\.nv\.constant)(\d+)(?:\.([^\s:]+))?", line)
        if header:
            section = (int(header[1]), header[2] or "")
            result.setdefault(section, {})
            continue
        words = re.match(r"\s*(?:0x)?([0-9a-f]{8,16})\s*:?\s+((?:0x[0-9a-f]{8}\s*)+)", line, re.I)
        if section and words:
            offset = int(words[1], 16)
            data = b"".join(struct.pack("<I", int(w, 16)) for w in words[2].split())
            for i, value in enumerate(data):
                key = offset + i
                if key in result[section] and result[section][key] != value:
                    raise Unmatched("conflicting ELF constant sections")
                result[section][key] = value
    return result


@dataclass(frozen=True)
class Instruction:
    address: int
    predicate: str
    opcode: str
    operands: tuple

    @property
    def destinations(self):
        if self.opcode.startswith(("ST", "BRA", "JMP", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "WARPSYNC", "MEMBAR", "DEPBAR", "NOP")):
            return ()
        if not WRITES.match(self.opcode):
            raise Unmatched(f"unknown destination semantics: {self.opcode}")
        count = 2 if self.opcode.startswith(("ISETP", "FSETP", "DSETP", "PSETP", "UISETP")) else 1
        return tuple(r for operand in self.operands[:count] for r in REG.findall(operand))


def instructions(text):
    parsed = parse_sass(text)
    addresses, name = {}, None
    for line in text.splitlines():
        function = re.match(r"\s*Function\s*:\s*(\S+)", line)
        if function:
            name = function[1]
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


class Slice:
    def __init__(self, functions, const, name, gf, gf_after):
        self.functions, self.const, self.name = functions, const, name
        self.gf, self.gf_after = gf, gf_after
        self.resolved, self.active, self.callers = {}, set(), {}

    def operand(self, operand, name, before):
        def constant(match):
            bank, offset = (int(x, 16) for x in match.groups())
            if bank == 0:
                raise Unmatched("c[0x0] kernel parameter")
            section = self.const.get((bank, name), self.const.get((bank, ""), {}))
            width = 8 if self.functions[name][before].opcode.startswith("D") else 4
            try:
                data = bytes(section[offset + i] for i in range(width))
            except KeyError as error:
                raise Unmatched(f"unresolved constant: {match[0]} in {name}") from error
            self.resolved[f"{name}:{match[0]}"] = data.hex()
            return "constant:" + data.hex()
        operand = CONST.sub(constant, operand)
        if "[" in operand or "]" in operand:
            raise Unmatched(f"indirect address: {operand}")
        pieces, end = [], 0
        for match in REG.finditer(operand):
            pieces.extend((operand[end:match.start()], self.register(match[0], name, before)))
            end = match.end()
        pieces.append(operand[end:])
        return tuple(pieces)

    def register(self, reg, name, before):
        if reg in ("RZ", "URZ", "PT", "UPT"):
            return ("fixed", reg)
        if name == self.name and reg == self.gf and before > self.gf_after:
            # A later write to the same physical register is a different value.
            if not any(reg in i.destinations for i in self.functions[name][self.gf_after + 1:before]):
                return ("gf",)
        key = name, before, reg
        if key in self.active:
            raise Unmatched("cyclic dependency")
        self.active.add(key)
        try:
            sequence = self.functions[name]
            for index in range(before - 1, -1, -1):
                ins = sequence[index]
                if ins.opcode.startswith(("CALL", "JCAL")):
                    target = ins.operands[-1].strip("`'() ")
                    if target not in self.functions:
                        raise Unmatched(f"unresolved call target: {target}")
                    callee = self.functions[target]
                    if any(reg in op.destinations for op in callee):
                        returns = [pc for pc, op in enumerate(callee) if op.opcode.startswith("RET")]
                        if len(returns) != 1:
                            raise Unmatched("subroutine must have one resolved return")
                        self.callers[target] = (name, index)
                        try:
                            value = self.register(reg, target, returns[0])
                        finally:
                            self.callers.pop(target)
                        return (ins.opcode, self.operand(ins.predicate[1:], name, index) if ins.predicate else (), value)
                if reg not in ins.destinations:
                    continue
                if ins.opcode.startswith(("LDG", "LDL", "LDS", "LDC", "ULDC", "S2R", "CS2R")):
                    # Direct constant loads can be resolved. All other memory inputs are unknown.
                    if ins.opcode.startswith(("LDC", "ULDC")) and all("[" not in CONST.sub("", x) for x in ins.operands[1:]):
                        pass
                    else:
                        raise Unmatched(f"unresolved load: {ins.opcode}")
                slot = ins.destinations.index(reg)
                operands = tuple(self.operand(x, name, index) for x in ins.operands[len(ins.destinations):])
                predicate = self.operand(ins.predicate.removeprefix("@"), name, index) if ins.predicate else ()
                previous = self.register(reg, name, index) if ins.predicate else ()
                # Keep branch conditions in the expression region. Unknown control is refused.
                branches = []
                for pc in range(index, before):
                    branch = sequence[pc]
                    if branch.opcode.startswith(("BRA", "JMP", "BRX")):
                        if not branch.predicate:
                            raise Unmatched("unresolved unconditional control flow")
                        branches.append((branch.opcode, self.operand(branch.predicate[1:], name, pc), branch.operands))
                return (ins.opcode, slot, operands, predicate, previous, tuple(branches))
            if name in self.callers:
                caller, call_index = self.callers[name]
                return self.register(reg, caller, call_index)
            raise Unmatched(f"undefined operand {reg}")
        finally:
            self.active.remove(key)

    def root(self, index, reg):
        # Linear def-use is valid only when control does not skip a definition.
        # Refuse branch joins until their reaching definitions are established.
        names = {self.name}
        pending = [self.name]
        while pending:
            name = pending.pop()
            for ins in self.functions[name][self.gf_after + 1:index + 1] if name == self.name else self.functions[name]:
                if ins.opcode.startswith(("BRA", "JMP", "BRX", "JMX")):
                    raise Unmatched("control-flow join has no proven reaching definitions")
                if ins.opcode.startswith(("CALL", "JCAL")):
                    target = ins.operands[-1].strip("`'() ")
                    if target not in self.functions:
                        raise Unmatched(f"unresolved call target: {target}")
                    if target not in names:
                        names.add(target)
                        pending.append(target)
        return self.register(reg, self.name, index + 1)


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


def prove(library_sass, library_elf, probe_sass, probe_elf):
    """Require every mode-0 expression to match, without register-name matching."""
    result = {"mode0_instantiations": 0, "quotients": 0, "constant_operands": {}, "unresolved": [], "matched": False}
    try:
        functions, probe = instructions(library_sass), instructions(probe_sass)
        pconst, lconst = constants(probe_elf), constants(library_elf)
        pname, = [name for name in probe if "swiglu_probe" in name]
        pseq = probe[pname]
        loads = [(i, r) for i, ins in enumerate(pseq) if ins.opcode.startswith("LDG") for r in ins.destinations]
        candidates = []
        for i, ins in enumerate(pseq):
            if not ins.opcode.startswith(("FMUL", "FFMA")):
                continue
            for dest in ins.destinations:
                if not reaches_store(pseq, i, dest):
                    continue
                for boundary, reg in loads:
                    if boundary < i:
                        sliced = Slice(probe, pconst, pname, reg, boundary)
                        try:
                            graph = sliced.root(i, dest)
                            if "MUFU" in str(graph) and "RCP" in str(graph) and "EX2" in str(graph):
                                candidates.append((graph, sliced.resolved))
                        except Unmatched:
                            pass
        if len(candidates) != 1:
            raise Unmatched(f"probe quotient candidates: {len(candidates)}")
        expected, resolved = candidates[0]
        result["constant_operands"].update(resolved)
        names = [name for name in functions if re.search(r"routed_fused_(?:fp4_)?kernelI(?:Lb[01]E)?Li0E", name)]
        if not names:
            raise Unmatched("no mode-0 instantiations")
        result["mode0_instantiations"] = len(names)
        for name in names:
            seq, found = functions[name], 0
            minima = [(i, r) for i, ins in enumerate(seq) if ins.opcode.startswith("FMNMX") for r in ins.destinations]
            roots = [(i, r) for i, ins in enumerate(seq) if ins.opcode.startswith(("FMUL", "FFMA")) for r in ins.destinations
                     if reaches_store(seq, i, r)]
            # Every exponential region must have a matched quotient reaching an act store.
            exponentials = sum(ins.opcode.startswith("MUFU.EX2") for ins in seq)
            for i, reg in roots:
                for boundary, gf in minima:
                    if boundary >= i:
                        continue
                    sliced = Slice(functions, lconst, name, gf, boundary)
                    try:
                        graph = sliced.root(i, reg)
                    except Unmatched:
                        continue
                    if graph == expected:
                        found += 1
                        result["constant_operands"].update(sliced.resolved)
                        break
            result["quotients"] += found
            if found != exponentials or not found:
                raise Unmatched(f"{name}: {found} quotients matched for {exponentials} exponential regions")
        result["matched"] = True
    except (Unmatched, ValueError) as error:
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
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    write_json(args.out, {r["key"]: match_library(r) for r in json.loads(args.build.read_text())["libraries"]})
