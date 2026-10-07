"""Non-serving semantic validation and admission of measured rung tables.

Validation returns its input or raises ValueError. Admission returns a decision
with status allow, wait, hold, excluded, unsupported or failed. Missing evidence
waits; this table never replaces export or serving gates.
"""
from __future__ import annotations

import math
import json
from datetime import datetime
from pathlib import PurePosixPath
import re

STATUSES = frozenset(("pending", "measured", "unsupported", "failed"))


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _text(value):
    return isinstance(value, str) and bool(value)


def _paired_mean_matches(measurement):
    """Only binary64 rounding from unit conversion/addition may move the mean."""
    a, b = measurement["pass_times_us"]
    expected = 0.5 * a + 0.5 * b
    # The harness multiplies its millisecond mean by 1000; stored pass values
    # multiply first. This bound covers those binary64 representation steps,
    # not a benchmark tolerance or a permission to change the statistic.
    roundoff = math.ulp(a) + math.ulp(b) + math.ulp(expected)
    _require(abs(measurement["kernel_time_us"] - expected) <= roundoff,
             "kernel time differs from mean of paired pass medians")





def _cell(cell):
    _require(isinstance(cell, dict), "cell must be an object")
    _require(_text(cell.get("cell_id")) and _text(cell.get("shape_id")), "cell identifiers missing")
    _require(cell.get("kernel_kind") in ("routed", "dense"), "unknown kernel kind")
    _require(_integer(cell.get("M")) and cell["M"] > 0, "invalid M")
    return tuple(cell[k] for k in ("cell_id", "kernel_kind", "shape_id", "M"))


def _geometry_v1(geom):
    bits = geom["bits_per_256_weight_tile"]
    decode = geom["decode_width"]
    rates = decode.get("run_widths")
    _require(isinstance(rates, list) and rates and all(_integer(v) and v > 0 for v in rates), "decode run widths missing")
    _require(all(_integer(decode.get(k)) and decode[k] > 0 for k in ("window_bits", "value_bits", "word_stages", "superblock_rows", "k_split")), "decode width fields missing")
    alignment = geom["alignment"]
    _require(_integer(alignment.get("slot_words")) and alignment["slot_words"] > 0, "alignment slot missing")
    for field in ("lane_bits", "lane_ends_on_word", "half_bytes", "half_copy"):
        _require(isinstance(alignment.get(field), list) and len(alignment[field]) == len(rates), "alignment fields missing")
    _require(all(_integer(v) and v > 0 for v in alignment["lane_bits"] + alignment["half_bytes"]), "alignment bit/byte widths")
    _require(all(isinstance(v, bool) for v in alignment["lane_ends_on_word"]) and all(_text(v) for v in alignment["half_copy"]), "alignment facts")

    _require(_integer(bits.get("numerator")) and bits["numerator"] > 0 and _integer(bits.get("denominator")) and bits["denominator"] > 0, "tile-bit rational")
    sm = geom["shared_memory"]
    _require(_integer(sm.get("requested_bytes")) and _integer(sm.get("available_bytes")) and 0 < sm["requested_bytes"] <= sm["available_bytes"] and sm.get("fits") is True, "shared-memory fit")
    reg = geom["register_pressure"]
    _require(_integer(reg.get("REG")) and reg["REG"] > 0 and all(_integer(reg.get(k)) and reg[k] >= 0 for k in ("STACK", "LOCAL", "SHARED")), "compiler resources missing")


def _geometry_v2(geom):
    """Explicit grammar variants; no absent WINDOW facts become positive numbers."""
    body,kind=geom.get('body_kind'),geom.get('decoder_kind')
    owners={'fused_window':('tessera.routed_fused',),'compact_window':('tessera.window_gemm','tessera.window_gemm_grouped'),
            'native_tcq':('tessera.kernel_a4',)}
    scopes={'fused_window':'raw_packed_window','compact_window':'compact_packed_window',
            'native_tcq':'native_tcq_decode_gemm'}
    _require(body in ('window','tcq') and kind in owners,'unknown body/decoder')
    _require((body=='window')==(kind in ('fused_window','compact_window')),'body/decoder disagreement')
    _require(geom.get('decoder_owner') in owners[kind] and geom.get('execution_scope')==scopes[kind],'decoder owner/execution scope')
    ring=geom.get('word_ring',{})
    _require(ring.get('owner')==geom.get('decoder_owner') and ring.get('kind')==('staged' if kind=='fused_window' else 'none'),'word-ring owner/facts')
    if kind=='fused_window':
        _geometry_v1(geom)
        _require(geom['shared_memory'].get('kind')=='used','fused WINDOW needs its shared-memory request')
        _require(geom['register_pressure'].get('compiler')=='cuda_cuobjdump','fused WINDOW compiler owner')
        return
    decode,alignment,sm,reg=(geom[k] for k in ('decode_width','alignment','shared_memory','register_pressure'))
    rates=decode.get('run_widths')
    _require(isinstance(rates,list) and rates and all(_integer(r) and r>0 for r in rates),'decode run widths missing')
    _require(all(_integer(decode.get(k)) and decode[k]>0 for k in ('value_bits','arity')),'decode value width/arity')
    _require('word_stages' in decode and decode['word_stages'] is None and 'slot_words' in alignment and alignment['slot_words'] is None,'non-ring decoder must declare absent WINDOW stages/slots')
    if body=='window':
        _require(_integer(decode.get('window_bits')) and decode['window_bits']>0 and max(rates)<=decode['window_bits'],'WINDOW width')
        _require(alignment.get('kind')=='column_chunk_words' and alignment.get('owner')=='tessera.kernel_window_gemv.Repacked','compact WINDOW layout owner')
        _require(alignment.get('word_alignment_bytes')==4 and alignment.get('tile_rows')==512 and _integer(alignment.get('tile_words')) and alignment['tile_words']>0,'compact WINDOW alignment')
        _require(alignment.get('column_chunk_words')==[16*r for r in rates],'compact WINDOW column chunks')
        _require(all(_integer(decode.get(k)) and decode[k]>0 for k in ('block_m','block_n','block_k')),'compact WINDOW launch blocks')
    else:
        _require(_integer(decode.get('window_bits')) and decode['window_bits']==0,'TCQ has no WINDOW body')
        _require(_integer(decode.get('memory')) and 0<decode['memory']<=8 and decode.get('span')==2,'TCQ span/history')
        _require(decode.get('history_lookup_bits')==decode['memory']+1 and decode.get('label_lut_entries')==1<<(decode['memory']+1),'TCQ history lookup facts')
        _require(max(rates)<=decode['value_bits']*decode['arity']-1,'TCQ code-rate cap')
        if kind=='native_tcq':
            _require(decode['arity']==2 and decode['value_bits']==4 and len(rates)==1,'native span-two E2M1 tuple/rate')
            _require(all(_integer(decode.get(k)) and decode[k]>0 for k in ('block_m','block_n','block_k','mma_k','scale_group')) and decode['mma_k']==64 and decode['scale_group']==16 and decode['block_k']%decode['mma_k']==0,'native TCQ launch geometry')
            _require(alignment.get('kind')=='tcq_planes' and alignment.get('owner')=='tessera.compact_prep.prepare_span2_compact','TCQ plane owner')
            shapes,byte_counts,widths=(alignment.get(k) for k in ('plane_shapes','plane_bytes','plane_element_bytes'))
            # Actual native ABI: A4Unit.from_prepared/_unit_dtype address byte
            # planes; label_lut is int32. These are element bits, not dummy sizes.
            element_bits={'select':8,'label':8,'point':8,'nibbles':8,'lut_bytes':8,'label_lut':32,'code_nibbles':8}
            _require(all(isinstance(v,dict) and set(v)==set(element_bits) for v in (shapes,byte_counts,widths)),'native TCQ mandatory plane census')
            for name,shape in shapes.items():
                _require(isinstance(shape,list) and shape and all(_integer(v) and v>=0 for v in shape),'TCQ plane shape')
                _require(_integer(widths[name]) and widths[name]*8==element_bits[name],'TCQ actual element width')
                _require(_integer(byte_counts[name]) and byte_counts[name]>=0 and math.prod(shape)*widths[name]==byte_counts[name],'TCQ exact shape/element byte count')
                _require(math.prod(shape)>0 or name=='point','mandatory native non-POINT plane is empty')
            _require(math.prod(shapes['lut_bytes'])==16 and math.prod(shapes['label_lut'])==decode['label_lut_entries'] and math.prod(shapes['code_nibbles'])==4*(1<<(rates[0]-1)),'native TCQ lookup plane shapes')
            _require((math.prod(shapes['point'])==0)==(rates[0]==1),'POINT applicability differs from actual field width')
    requested,available=sm.get('requested_bytes'),sm.get('available_bytes')
    _require(_integer(available) and available>0 and _integer(requested) and requested<=available and sm.get('fits') is True,'shared-memory fit/facts')
    _require((sm.get('kind')=='used' and requested>0) or (sm.get('kind')=='none' and requested==0),'explicit shared-memory presence')
    _require(_integer(reg.get('REG')) and reg['REG']>0,'register pressure missing')
    if reg.get('compiler')=='triton_compiled_kernel':
        _require(_integer(reg.get('SPILLS')) and reg['SPILLS']>=0 and _integer(reg.get('SHARED')) and reg['SHARED']==requested and _text(reg.get('compiler_symbol')),'actual Triton compiler resources')
        _require('STACK' in reg and reg['STACK'] is None and 'LOCAL' in reg and reg['LOCAL'] is None,'Triton does not report CUDA stack/local byte counts')
    else:
        _require(reg.get('compiler')=='cuda_cuobjdump' and all(_integer(reg.get(k)) and reg[k]>=0 for k in ('STACK','LOCAL','SHARED')),'actual CUDA compiler resources')
    bits=geom['bits_per_256_weight_tile']
    _require(_integer(bits.get('numerator')) and bits['numerator']>0 and _integer(bits.get('denominator')) and bits['denominator']>0,'tile-bit rational')



def _quality_scope_v2(quality,measurements,format_name,rung):
    scope=quality.get('scope')
    _require(isinstance(scope,dict) and scope.get('owner')=='tessera.export.encode_linear','quality producer scope missing')
    match=re.fullmatch(r'TESSERA_([A-Z0-9]+)_K(\d+)',format_name)
    _require(match is not None,'quality family format')
    base,arity=match[1],int(match[2])
    expected_grid=base if arity==1 else base+'x'+str(arity)
    _require(scope.get('format')==format_name and scope.get('grid')==expected_grid and _integer(scope.get('arity')) and scope['arity']==arity and scope.get('rung')==rung,'quality format/grid/arity/rung scope')
    recipe=scope.get('recipe')
    _require(isinstance(recipe,dict) and set(recipe)=={'body','span','plane','window_bits','seed','sigma','channel_sigma'},'quality encoder recipe scope')
    kinds=scope.get('kernel_kinds')
    _require(isinstance(kinds,list) and kinds and set(kinds)<= {'dense','routed'},'quality structure scope')
    for m in measurements:
        g=m['geometry']
        _require(m['kernel_kind'] in kinds and g.get('recipe')==recipe and recipe.get('body')==g.get('body_kind'),'quality/measurement recipe or structure mismatch')
        _require(g['decode_width'].get('arity',1)==arity,'quality/measurement arity mismatch')



def validate_index(index):
    """Validate an explicitly versioned index; paths are index-root relative."""
    schema=INDEX_SCHEMAS.get(index.get('schema')) if isinstance(index,dict) and _text(index.get('schema')) else None
    _require(schema is not None,'index schema')
    _structure(index,schema)
    _require(isinstance(index.get("formats"), dict), "index formats")
    for fmt, entry in index["formats"].items():
        _require(_text(fmt) and isinstance(entry, dict) and isinstance(entry.get("kernel_builds"), dict), "format entry")
        for build, item in entry["kernel_builds"].items():
            _require(_text(build) and re.fullmatch(r"[A-Za-z0-9_.-]+", build), "build identifier")
            version = item.get("current_version")
            versions = item.get("versions")
            _require(_integer(version) and version >= 1 and isinstance(versions, dict) and str(version) in versions, "current version missing")
            for number, rec in versions.items():
                _require(isinstance(number, str) and number.isdigit() and str(int(number)) == number and int(number) >= 1, "version key")
                _require(isinstance(rec, dict), "version entry")
                path = rec.get("path")
                _require(_text(path) and "\\" not in path and not PurePosixPath(path).is_absolute() and all(p not in ("", ".", "..") for p in path.split("/")), "unsafe table path")
                allowed=('fleet.rung_allowability.v1',) if index['schema']=='fleet.rung_allowability.index.v1' else tuple(TABLE_SCHEMAS)
                _require(rec.get('table_schema') in allowed and rec.get('table_status') in ('partial','complete'),'table schema/status')
    return index


def validate_table(table):
    """Validate published structure and measured-row semantics; invalid data refuses."""
    schema=TABLE_SCHEMAS.get(table.get('schema')) if isinstance(table,dict) and _text(table.get('schema')) else None
    _require(schema is not None,'table schema')
    _structure(table,schema)
    _require(_integer(table.get("table_version")) and table["table_version"] >= 1, "table version")
    _require(table.get("table_status") in ("partial", "complete") and _text(table.get("format")), "table status/format")
    build = table.get("kernel_build", {})
    _require(all(_text(build.get(k)) for k in ("id", "source_commit", "library_variant", "architecture", "activation_contract")), "build context missing")
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+", build["id"]), "build identifier")
    scope = table.get("scope", {})
    lo, hi, step = (scope.get(k) for k in ("rung_min", "rung_max", "grid_step_q256"))
    _require(all(_integer(v) and v > 0 for v in (lo, hi, step)) and lo <= hi and (hi-lo) % step == 0, "rung grid")
    _require(_text(scope.get("grid_owner")), "grid owner")
    required = scope.get("required_cells")
    _require(isinstance(required, list) and required, "required cells missing")
    keys = [_cell(c) for c in required]
    _require(len(keys) == len(set(keys)) and len({k[0] for k in keys}) == len(keys), "duplicate required cell")
    rows = table.get("rungs")
    _require(isinstance(rows, list), "rung rows")
    by_rung = {}
    for row in rows:
        _require(isinstance(row, dict), "rung object")
        q = row.get("rung")
        _require(_integer(q) and lo <= q <= hi and (q-lo) % step == 0 and q not in by_rung, "duplicate/out-of-grid rung")
        by_rung[q] = row
        _require(row.get("measurement_status") in STATUSES and (row.get("supported") is None or isinstance(row["supported"], bool)), "rung status/support")
        _require(isinstance(row.get("anomaly_flags"), list) and all(_text(f) for f in row["anomaly_flags"]) and len(set(row["anomaly_flags"])) == len(row["anomaly_flags"]), "anomaly flags")
        _require(isinstance(row.get("observations"), list) and isinstance(row.get("quality"), dict) and isinstance(row.get("dominance_evidence"), list), "rung evidence fields")
        _require(isinstance(row.get("excluded"), bool), "exclusion flag")
        quality_flags = row["quality"].get("anomaly_flags", [])
        _require(isinstance(quality_flags, list) and all(_text(f) for f in quality_flags) and len(set(quality_flags)) == len(quality_flags), "quality anomaly flags")
        _require(set(quality_flags) == set(row["anomaly_flags"]), "quality and rung anomaly flags disagree")
        measurements = row.get("measurements")
        _require(isinstance(measurements, list), "measurements")
        seen = set()
        for m in measurements:
            key = _cell(m)
            _require(key in keys and key not in seen, "unknown/duplicate measurement cell")
            seen.add(key)
            _require(m.get("measurement_status") in STATUSES and isinstance(m.get("evidence"), dict), "measurement status/evidence")
            t = m.get("kernel_time_us")
            _require(t is None or _number(t), "invalid kernel time")
            if m["measurement_status"] == "measured":
                _require(_number(t) and t > 0 and _text(m.get("kernel_path")), "measured kernel time/path")
                _require(m.get("measurement_build_id") == build["id"], "measurement build scope")
                geom = m.get("geometry")
                _require(isinstance(geom, dict) and all(isinstance(geom.get(k), dict) for k in ("bits_per_256_weight_tile", "alignment", "shared_memory", "register_pressure", "decode_width")), "measured geometry missing")
                if table['schema']=='fleet.rung_allowability.v1':
                    _geometry_v1(geom)
                else:
                    _geometry_v2(geom)
                _require(isinstance(m.get("pass_times_us"), list) and len(m["pass_times_us"]) == 2 and all(_number(t) and t > 0 for t in m["pass_times_us"]), "paired pass times")
                _paired_mean_matches(m)
        if row["measurement_status"] == "measured":
            _require(row["supported"] is True and seen == set(keys) and all(m["measurement_status"] == "measured" for m in measurements), "measured row incomplete")
            quality = row["quality"]
            if table['schema']=='fleet.rung_allowability.v2':
                _quality_scope_v2(quality,measurements,table['format'],q)
            if table["schema"] != "fleet.rung_allowability.v3":
                _require(quality.get("measurement_status") == "measured" and quality.get("source_kind") == "actual_sampled_expert_weights" and quality.get("device") == "cpu" and quality.get("samples"), "CPU quality incomplete")
            for sample in quality.get("samples", []):
                _require(_number(sample.get("relative_sse")) and _number(sample.get("source_squared_norm")) and sample["source_squared_norm"] > 0 and _integer(sample.get("exact_bytes")) and sample["exact_bytes"] > 0 and _text(sample.get("source_sha256")), "quality sample evidence")
        else:
            _require(not row["excluded"], "unmeasured rung excluded")
        if not row["excluded"]:
            _require(row.get("dominating_rung") is None and not row["dominance_evidence"], "spurious dominance")
    _require(set(by_rung) == set(range(lo, hi+1, step)), "missing census rung")
    if table["table_status"] == "complete":
        _require(all(r["measurement_status"] != "pending" for r in rows), "complete table has pending rows")
    for row in rows:
        if not row["excluded"]:
            continue
        q = row["rung"]
        dq = row.get("dominating_rung")
        _require(_integer(dq) and q < dq <= q+step and dq in by_rung, "dominator not adjacent higher")
        higher = by_rung[dq]
        _require(higher["measurement_status"] == "measured" and higher["supported"] is True and not higher["anomaly_flags"], "dominator ineligible")
        evidence = row["dominance_evidence"]
        _require(len(evidence) == len(keys) and {e.get("cell_id") for e in evidence} == {k[0] for k in keys}, "dominance cell coverage")
        for e in evidence:
            canonical_low = next(m for m in row["measurements"] if m["cell_id"] == e["cell_id"])
            canonical_high = next(m for m in higher["measurements"] if m["cell_id"] == e["cell_id"])
            low = e.get("lower_measurement", canonical_low)
            high = e.get("higher_measurement", canonical_high)
            # Neighbor-overlap quanta may carry the same higher rung in two
            # paired runs. Its witness is real measured data, not a substituted
            # canonical timing from a different clock window.
            for witness, canonical in ((low, canonical_low), (high, canonical_high)):
                _structure(witness,schema['$defs']['measurement'],schema)
                _require(_cell(witness) == _cell(canonical), "dominance witness cell scope")
                _require(witness["measurement_status"] == "measured" and witness.get("measurement_build_id") == build["id"], "dominance witness build/status")
                _require(witness["geometry"] == canonical["geometry"] and witness["kernel_path"] == canonical["kernel_path"], "dominance witness kernel/geometry")
                _require(_number(witness["kernel_time_us"]) and witness["kernel_time_us"] > 0, "dominance witness time")
                _require(isinstance(witness.get("pass_times_us"), list) and len(witness["pass_times_us"]) == 2 and all(_number(t) and t > 0 for t in witness["pass_times_us"]), "dominance witness passes")
                _paired_mean_matches(witness)

            _require(e.get("lower_time_us") == low["kernel_time_us"] and e.get("higher_time_us") == high["kernel_time_us"] <= low["kernel_time_us"], "dominance times")
            for field in ("comparison_id", "paired_seed_contract", "timing_statistic", "timer"):
                _require(_text(low["evidence"].get(field)) and low["evidence"].get(field) == high["evidence"].get(field), "unpaired dominance: " + field)
            _require(e.get("comparison_id") == low["evidence"]["comparison_id"], "dominance lineage")
    if table["schema"] == "fleet.rung_allowability.v3":
        _require(table["geometry_classes"] == measured_geometry_classes(table), "class identity or observed timing scope differs")
        shapes = {(shape["kernel_kind"], shape["shape_id"]): (shape["rows"], shape["columns"]) for shape in scope.get("shapes", [])}
        for row in rows:
            for measurement in row["measurements"]:
                if measurement["measurement_status"] == "measured":
                    evidence = measurement["evidence"]
                    _require(shapes.get((measurement["kernel_kind"], measurement["shape_id"])) == (evidence.get("rows"), evidence.get("columns")), "measurement differs from declared shape")
    return table


def admit_rung(table, *, format, kernel_build_id, rung, scope=None, cell_ids=None, activation_contract=None, recipe=None):
    """Return a structured decision. Invalid evidence raises ValueError, not allow."""
    if table is None:
        return {"status": "wait", "reason": "missing_table", "rung": rung}
    validate_table(table)
    if table["format"] != format or table["kernel_build"]["id"] != kernel_build_id:
        return {"status": "wait", "reason": "unmeasured_format_or_build", "rung": rung}
    if scope is not None and any(table["scope"].get(k) != v for k, v in scope.items()):
        return {"status": "wait", "reason": "unmeasured_scope", "rung": rung}
    row = next((r for r in table["rungs"] if r["rung"] == rung), None)
    if table["schema"] == "fleet.rung_allowability.v3":
        return _admit_performance_scope(table, row, rung, cell_ids, activation_contract, recipe)
    if cell_ids is not None or activation_contract is not None or recipe is not None:
        return {"status": "wait", "reason": "scoped_policy_requires_v3_table", "rung": rung}
    if row is None or row["measurement_status"] == "pending":
        status, reason = "wait", "missing_measurement"
    elif row["measurement_status"] in ("unsupported", "failed"):
        status, reason = row["measurement_status"], row["measurement_status"]
    elif row["anomaly_flags"]:
        status, reason = "hold", "quality_or_correctness_anomaly"
    elif row["excluded"]:
        status, reason = "excluded", "adjacent_higher_all_cell_dominance"
    else:
        status, reason = "allow", "measured_supported_no_anomaly"
    return {"status": status, "reason": reason, "rung": rung, "table_version": table["table_version"], "kernel_build_id": kernel_build_id}

# These are the published structural schemas, not a reduced semantic mirror.
TABLE_SCHEMA = json.loads("{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"$id\":\"fleet.rung_allowability.v1\",\"title\":\"D41 measured rung allowability table\",\"type\":\"object\",\"required\":[\"schema\",\"table_version\",\"table_status\",\"format\",\"kernel_build\",\"generated_at\",\"scope\",\"rungs\"],\"properties\":{\"schema\":{\"const\":\"fleet.rung_allowability.v1\"},\"table_version\":{\"type\":\"integer\",\"minimum\":1},\"table_status\":{\"enum\":[\"partial\",\"complete\"]},\"format\":{\"type\":\"string\",\"minLength\":1},\"generated_at\":{\"type\":\"string\",\"format\":\"date-time\"},\"kernel_build\":{\"type\":\"object\",\"required\":[\"id\",\"source_commit\",\"library_variant\",\"architecture\",\"activation_contract\"],\"properties\":{\"id\":{\"type\":\"string\",\"pattern\":\"^[A-Za-z0-9_.-]+$\"},\"source_commit\":{\"type\":\"string\"},\"library_variant\":{\"type\":\"string\"},\"architecture\":{\"type\":\"string\"},\"activation_contract\":{\"type\":\"string\"},\"metadata\":{\"type\":\"object\"}},\"description\":\"Observed kernel/build and execution context, not a new run-identity seal. A decode-path change republishes a table with affected rows pending until remeasured. Unaffected rows may retain explicit measurement lineage.\"},\"scope\":{\"type\":\"object\",\"required\":[\"rung_min\",\"rung_max\",\"grid_step_q256\",\"grid_owner\",\"required_cells\"],\"properties\":{\"rung_min\":{\"type\":\"integer\",\"minimum\":1},\"rung_max\":{\"type\":\"integer\",\"minimum\":1},\"grid_step_q256\":{\"type\":\"integer\",\"minimum\":1},\"grid_owner\":{\"type\":\"string\"},\"required_cells\":{\"type\":\"array\",\"minItems\":1,\"items\":{\"$ref\":\"#/$defs/cell_key\"}},\"shapes\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"timing_statistic\":{\"type\":\"string\"}},\"description\":\"The scalar q256 domain and exact cell roster are derived from their owners. Initial T-8 scope is R768 through R1152 inclusive, true grid step 1, M 1/16/2048/4096 and 2-3 representative GLM shapes.\"},\"rungs\":{\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/rung\"}},\"evidence\":{\"type\":\"object\"}},\"$defs\":{\"status\":{\"enum\":[\"pending\",\"measured\",\"unsupported\",\"failed\"]},\"cell_key\":{\"type\":\"object\",\"required\":[\"cell_id\",\"kernel_kind\",\"shape_id\",\"M\"],\"properties\":{\"cell_id\":{\"type\":\"string\"},\"kernel_kind\":{\"enum\":[\"routed\",\"dense\"]},\"shape_id\":{\"type\":\"string\"},\"M\":{\"type\":\"integer\",\"minimum\":1}}},\"geometry\":{\"type\":\"object\",\"required\":[\"bits_per_256_weight_tile\",\"alignment\",\"shared_memory\",\"register_pressure\",\"decode_width\"],\"properties\":{\"bits_per_256_weight_tile\":{\"anyOf\":[{\"type\":\"null\"},{\"type\":\"object\",\"required\":[\"numerator\",\"denominator\"],\"properties\":{\"numerator\":{\"type\":\"integer\",\"minimum\":0},\"denominator\":{\"type\":\"integer\",\"minimum\":1}}}]},\"alignment\":{\"type\":[\"object\",\"null\"]},\"shared_memory\":{\"type\":[\"object\",\"null\"]},\"register_pressure\":{\"type\":[\"object\",\"null\"]},\"decode_width\":{\"type\":[\"object\",\"null\"]},\"raw\":{\"type\":\"object\"}},\"description\":\"Actual accountant/launch/compiler fields, per cell when shape or M changes them. Store exact rational tile bits; shared memory records requested/available bytes and fit; registers record compiler resources/spills; decode width records window/value bits, run widths and word stages. Null means missing evidence, never zero-cost evidence.\"},\"measurement\":{\"allOf\":[{\"$ref\":\"#/$defs/cell_key\"}],\"type\":\"object\",\"required\":[\"measurement_status\",\"kernel_time_us\",\"kernel_path\",\"geometry\",\"evidence\"],\"properties\":{\"measurement_status\":{\"$ref\":\"#/$defs/status\"},\"kernel_time_us\":{\"type\":[\"number\",\"null\"],\"minimum\":0},\"kernel_path\":{\"type\":[\"string\",\"null\"]},\"geometry\":{\"anyOf\":[{\"$ref\":\"#/$defs/geometry\"},{\"type\":\"null\"}]},\"evidence\":{\"type\":\"object\"},\"pass_times_us\":{\"type\":\"array\",\"items\":{\"type\":\"number\",\"minimum\":0}},\"measurement_build_id\":{\"type\":\"string\"}}},\"rung\":{\"type\":\"object\",\"required\":[\"rung\",\"measurement_status\",\"supported\",\"anomaly_flags\",\"observations\",\"excluded\",\"dominating_rung\",\"measurements\",\"quality\",\"dominance_evidence\"],\"properties\":{\"rung\":{\"type\":\"integer\",\"minimum\":1},\"measurement_status\":{\"$ref\":\"#/$defs/status\"},\"supported\":{\"type\":[\"boolean\",\"null\"]},\"anomaly_flags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"uniqueItems\":true},\"observations\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"excluded\":{\"type\":\"boolean\"},\"dominating_rung\":{\"type\":[\"integer\",\"null\"],\"minimum\":1},\"measurements\":{\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/measurement\"}},\"quality\":{\"type\":\"object\"},\"dominance_evidence\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"lineage\":{\"type\":\"object\"}},\"allOf\":[{\"if\":{\"properties\":{\"excluded\":{\"const\":true}}},\"then\":{\"properties\":{\"measurement_status\":{\"const\":\"measured\"},\"dominating_rung\":{\"type\":\"integer\",\"minimum\":1}}}}],\"description\":\"Unique scalar q256 rung. measured requires all required cells and geometry plus completed CPU quality screen. excluded is ONLY the measured D41 performance proof: a higher supported rung no farther than one grid step has time <= this rung in every relevant cell, with paired comparable evidence. Quality/correctness anomaly flags separately block production until explained; merely slow or missing-census observations do not fabricate performance exclusion. Missing/unmeasured rows wait. Reader admits only measured, supported, non-excluded rows with empty anomaly_flags, alongside existing serving/export gates. Semantic validation also checks uniqueness, full cell coverage, finite times, same format/build/scope, strict higher-rung/one-step relation and every dominance comparison.\"}}}")
INDEX_SCHEMA = json.loads("{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",\"$id\":\"fleet.rung_allowability.index.v1\",\"title\":\"Versioned measured rung allowability index\",\"type\":\"object\",\"required\":[\"schema\",\"formats\"],\"properties\":{\"schema\":{\"const\":\"fleet.rung_allowability.index.v1\"},\"formats\":{\"type\":\"object\",\"additionalProperties\":{\"type\":\"object\",\"required\":[\"kernel_builds\"],\"properties\":{\"kernel_builds\":{\"type\":\"object\",\"propertyNames\":{\"pattern\":\"^[A-Za-z0-9_.-]+$\"},\"additionalProperties\":{\"type\":\"object\",\"required\":[\"current_version\",\"versions\"],\"properties\":{\"current_version\":{\"type\":\"integer\",\"minimum\":1},\"versions\":{\"type\":\"object\",\"propertyNames\":{\"pattern\":\"^[1-9][0-9]*$\"},\"additionalProperties\":{\"type\":\"object\",\"required\":[\"path\",\"table_schema\",\"table_status\"],\"properties\":{\"path\":{\"type\":\"string\",\"minLength\":1},\"table_schema\":{\"const\":\"fleet.rung_allowability.v1\"},\"table_status\":{\"enum\":[\"partial\",\"complete\"]}}}}}}}}}}},\"description\":\"Canonical semantic validation is tessera.rung_allowability.validate_index; current_version must select an existing version and table paths must be safe index-root-relative paths.\"}")


def _structure(value, schema, root=None, path="$ "):
    """Evaluate the structural vocabulary used by the two published schemas."""
    root = schema if root is None else root
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        _structure(value, target, root, path)
    for rule in schema.get("allOf", []):
        _structure(value, rule, root, path)
    if "anyOf" in schema:
        for rule in schema["anyOf"]:
            try:
                _structure(value, rule, root, path)
                break
            except ValueError:
                pass
        else:
            raise ValueError(path + ": no schema alternative matches")
    if "if" in schema:
        try:
            _structure(value, schema["if"], root, path)
        except ValueError:
            pass
        else:
            _structure(value, schema.get("then", {}), root, path)
    types = schema.get("type")
    if types:
        types = types if isinstance(types, list) else [types]
        valid = {"object":isinstance(value,dict), "array":isinstance(value,list),
                 "string":isinstance(value,str), "integer":_integer(value),
                 "number":_number(value), "boolean":isinstance(value,bool), "null":value is None}
        _require(any(valid[t] for t in types), path + ": schema type")
    if "const" in schema:
        _require(value == schema["const"], path + ": schema const")
    if "enum" in schema:
        _require(value in schema["enum"], path + ": schema enum")
    if isinstance(value, dict):
        _require(all(k in value for k in schema.get("required", [])), path + ": missing required schema key")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if "propertyNames" in schema:
                _structure(key, schema["propertyNames"], root, path + ".<key>")
            if key in properties:
                _structure(item, properties[key], root, path + "." + key)
            elif isinstance(schema.get("additionalProperties"), dict):
                _structure(item, schema["additionalProperties"], root, path + "." + key)
            elif schema.get("additionalProperties") is False:
                raise ValueError(path + ": extra schema key " + key)
    if isinstance(value, list):
        _require(len(value) >= schema.get("minItems", 0), path + ": schema minimum items")
        if schema.get("uniqueItems"):
            _require(all(item not in value[:i] for i,item in enumerate(value)), path + ": duplicate schema item")
        for i,item in enumerate(value):
            if "items" in schema:
                _structure(item, schema["items"], root, path + f"[{i}]")
    if isinstance(value,str):
        _require(len(value) >= schema.get("minLength",0), path + ": schema minimum length")
        if "pattern" in schema:
            _require(re.search(schema["pattern"],value), path + ": schema pattern")
        if schema.get("format") == "date-time":
            _require(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})",value), path + ": schema date-time")
            try:
                datetime.fromisoformat(value.upper().replace("Z","+00:00"))
            except ValueError as exc:
                raise ValueError(path + ": schema date-time") from exc
    if "minimum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        _require(_number(value) and value >= schema["minimum"], path + ": schema minimum")




# V1 objects and their strict WINDOW semantics are retained verbatim above.
# V2 is an explicit new grammar, never a guessed fallback for an old table.
import copy as _copy
TABLE_SCHEMA_V2=_copy.deepcopy(TABLE_SCHEMA)
TABLE_SCHEMA_V2['$id']='fleet.rung_allowability.v2'
TABLE_SCHEMA_V2['properties']['schema']={'const':'fleet.rung_allowability.v2'}
_geometry_schema=TABLE_SCHEMA_V2['$defs']['measurement']['properties']['geometry']
_geometry_schema.update(type='object',required=['bits_per_256_weight_tile','alignment','shared_memory','register_pressure','decode_width'],
                        properties={name:{'type':['object','null'] if name=='register_pressure' else 'object'} for name in ('bits_per_256_weight_tile','alignment','shared_memory','register_pressure','decode_width')})
_geometry_schema['required']+=['body_kind','decoder_kind','decoder_owner','execution_scope','word_ring']
_geometry_schema['properties'].update({
    'body_kind':{'enum':['window','tcq']},
    'decoder_kind':{'enum':['fused_window','compact_window','native_tcq']},
    'decoder_owner':{'type':'string','minLength':1},
    'execution_scope':{'enum':['raw_packed_window','compact_packed_window','native_tcq_decode_gemm']},
    'word_ring':{'type':'object','required':['kind','owner'],'properties':{'kind':{'enum':['staged','none']},'owner':{'type':'string','minLength':1}}},
})
_geometry_schema['allOf']=[{'if':{'properties':{'body_kind':{'const':'window'}}},'then':{'properties':{'decode_width':{'required':['window_bits'],'properties':{'window_bits':{'type':'integer','minimum':1}}}}}},
    {'if':{'properties':{'body_kind':{'const':'tcq'}}},'then':{'properties':{'decode_width':{'required':['window_bits','memory','span','history_lookup_bits','label_lut_entries'],'properties':{'window_bits':{'const':0}}}}}}]
INDEX_SCHEMA_V2=_copy.deepcopy(INDEX_SCHEMA)
INDEX_SCHEMA_V2['$id']='fleet.rung_allowability.index.v2'
INDEX_SCHEMA_V2['properties']['schema']={'const':'fleet.rung_allowability.index.v2'}
_version_schema=INDEX_SCHEMA_V2['properties']['formats']['additionalProperties']['properties']['kernel_builds']['additionalProperties']['properties']['versions']['additionalProperties']
_version_schema['properties']['table_schema']={'enum':['fleet.rung_allowability.v1','fleet.rung_allowability.v2']}
TABLE_SCHEMAS={'fleet.rung_allowability.v1':TABLE_SCHEMA,'fleet.rung_allowability.v2':TABLE_SCHEMA_V2}
INDEX_SCHEMAS={'fleet.rung_allowability.index.v1':INDEX_SCHEMA,'fleet.rung_allowability.index.v2':INDEX_SCHEMA_V2}



PERFORMANT_POLICY = {"kind": "whole_bit_per_structure", "authority": "Latest Rob/CEO D41 ruling, 2026-10-06",
                     "qualification_scope": "performance evidence only; export, numerical and serving gates remain independent"}


def performant_rungs(format, kernel_kind):
    """The owning menu, not a consumer-side copy or an encoder-capacity guess."""
    _require(kernel_kind in ("dense", "routed"), "unknown performance structure")
    if format == "TESSERA_E4M3_K1":
        return (768, 1024)
    if format == "TESSERA_BF16_K1":
        return tuple(range(256, (3584 if kernel_kind == "dense" else 2048) + 1, 256))
    if format == "TESSERA_E2M1_K2":
        return ()
    return ()


def scope_cell_ids(table, *, kernel_kind, rows, columns, M, routing=None):
    """Resolve actual declared shapes. A routed shape never stands in for dense."""
    validate_table(table)
    _require(kernel_kind in ("dense", "routed") and all(_integer(value) and value > 0 for value in (rows, columns, M)), "invalid requested shape/M scope")
    matches = {shape['shape_id'] for shape in table['scope'].get('shapes', [])
               if shape['kernel_kind'] == kernel_kind and shape['rows'] == rows and shape['columns'] == columns}
    return tuple(cell['cell_id'] for cell in table['scope']['required_cells']
                 if cell['kernel_kind'] == kernel_kind and cell['shape_id'] in matches and cell['M'] == M
                 and (routing is None or cell.get('routing', 'balanced' if kernel_kind == 'routed' else 'none') == routing))


def geometry_class_identity(table, rung, measurement):
    """Classify the actual decoder. Tuple arity is symbol rate, not scalar rate."""
    match = re.fullmatch(r'TESSERA_([A-Z0-9]+)_K(\d+)', table['format'])
    _require(match is not None, 'class family')
    arity = int(match[2])
    low, remainder = divmod(rung * arity, 256)
    geometry = measurement['geometry']
    decode = geometry['decode_width']
    widths = [low, low + 1] if remainder else [low]
    _require(decode.get('arity', 1) == arity and sorted(set(decode['run_widths'])) == widths, 'class actual rate/arity')
    _require(decode.get("value_bits") == {"BF16": 16, "E4M3": 8, "E2M1": 4}.get(match[1]), "class payload width differs from family")
    evidence = measurement['evidence']
    _require(all(_integer(evidence.get(key)) and evidence[key] > 0 for key in ('rows', 'columns')), 'class actual shape')
    _require(isinstance(geometry.get('recipe'), dict), 'class actual recipe')
    return {'format': table['format'], 'arity': arity, 'kind': 'mixed' if remainder else 'pure', 'run_widths': widths,
            'kernel_build_id': table['kernel_build']['id'], 'activation_contract': table['kernel_build']['activation_contract'],
            'cell_id': measurement['cell_id'], 'kernel_kind': measurement['kernel_kind'], 'M': measurement['M'],
            'shape': [evidence['rows'], evidence['columns']], 'routing': evidence.get('routing'),
            "mode": evidence.get("mode"), "epilogue": evidence.get("epilogue"),
            "input_distribution": evidence.get("input_distribution"),
            "alignment": {key: value for key, value in geometry["alignment"].items() if key not in ("tile_words", "plane_shapes", "plane_bytes")},
            'recipe': geometry['recipe'], 'decoder_kind': geometry['decoder_kind'], 'decoder_owner': geometry['decoder_owner'],
            'execution_scope': geometry['execution_scope'], 'kernel_path': measurement['kernel_path'],
            'decode_width': decode, 'shared_memory': geometry['shared_memory'], 'register_pressure': geometry['register_pressure']}


def measured_geometry_classes(table):
    groups = {}
    for row in table['rungs']:
        for measurement in row['measurements']:
            if measurement['measurement_status'] != 'measured':
                continue
            identity = geometry_class_identity(table, row['rung'], measurement)
            key = json.dumps(identity, sort_keys=True, separators=(',', ':'))
            group = groups.setdefault(key, {'identity': identity, 'observed_rungs': []})
            group['observed_rungs'].append(row['rung'])
    return [groups[key] for key in sorted(groups)]


def _admit_performance_scope(table, row, rung, cell_ids, activation_contract, recipe):
    required = {cell['cell_id']: cell for cell in table['scope']['required_cells']}
    selected = tuple(required) if cell_ids is None else tuple(cell_ids)
    if not selected or any(cell_id not in required for cell_id in selected):
        return {'status': 'wait', 'reason': 'unmeasured_shape_or_M_scope', 'rung': rung}
    if activation_contract is not None and table['kernel_build']['activation_contract'] != activation_contract:
        return {'status': 'wait', 'reason': 'unmeasured_activation_scope', 'rung': rung}
    cells = {measurement['cell_id']: measurement for measurement in row['measurements']} if row is not None else {}
    results = []
    for cell_id in selected:
        kind = required[cell_id]['kernel_kind']
        if rung not in performant_rungs(table['format'], kind):
            status = 'wait' if table['format'] == 'TESSERA_E2M1_K2' else 'excluded'
            results.append({'cell_id': cell_id, 'status': status, 'reason': 'performance_admission_not_established' if status == 'wait' else 'outside_performant_menu'})
            continue
        measurement = cells.get(cell_id)
        if measurement is None or measurement['measurement_status'] == 'pending':
            results.append({'cell_id': cell_id, 'status': 'wait', 'reason': 'missing_actual_measurement'})
        elif measurement['measurement_status'] != 'measured':
            results.append({'cell_id': cell_id, 'status': measurement['measurement_status'], 'reason': measurement['measurement_status']})
        elif row["supported"] is False or row["measurement_status"] in ("failed", "unsupported"):
            state = row["measurement_status"] if row["measurement_status"] in ("failed", "unsupported") else "unsupported"
            results.append({"cell_id": cell_id, "status": state, "reason": "recorded_source_refusal"})
        elif recipe is not None and measurement['geometry'].get('recipe') != recipe:
            results.append({'cell_id': cell_id, 'status': 'wait', 'reason': 'unmeasured_recipe_scope'})
        elif row['anomaly_flags']:
            results.append({'cell_id': cell_id, 'status': 'hold', 'reason': 'recorded_correctness_hold'})
        else:
            results.append({'cell_id': cell_id, 'status': 'allow', 'reason': 'measured_performant_scope'})
    status = next((state for state in ('failed', 'unsupported', 'hold', 'excluded', 'wait') if any(result['status'] == state for result in results)), 'allow')
    return {'status': status, 'reason': 'per_cell_performance_policy', 'rung': rung, 'cells': results,
            'table_version': table['table_version'], 'kernel_build_id': table['kernel_build']['id'],
            'numerical_qualification_inherited': False, 'serving_qualification_inherited': False}


def rung_speed(table, *, rung, cell_id=None, class_identity=None):
    """Actual timing or explicit class-derived timing, never inherited admission."""
    validate_table(table)
    if not _integer(rung) or not table["scope"]["rung_min"] <= rung <= table["scope"]["rung_max"]:
        return {"status": "wait", "reason": "outside_declared_rate_scope"}
    row = next((row for row in table['rungs'] if row['rung'] == rung), None)
    selected_id = cell_id if cell_id is not None else (class_identity or {}).get('cell_id')
    measurement = next((cell for cell in row['measurements'] if cell['cell_id'] == selected_id), None) if row else None
    if class_identity is not None and selected_id != class_identity.get("cell_id"):
        return {"status": "wait", "reason": "different_requested_cell_scope"}
    if measurement is not None and class_identity is not None and geometry_class_identity(table, rung, measurement) != class_identity:
        return {"status": "wait", "reason": "different_geometry_class"}
    if row is not None and row["anomaly_flags"]:
        return {"status": "hold", "reason": "recorded_correctness_hold"}
    if measurement is not None and measurement['measurement_status'] == 'measured':
        return {'status': 'measured', 'rung': rung, 'measurement': measurement,
                'menu_admitted': rung in performant_rungs(table['format'], measurement['kernel_kind']),
                'numerical_qualification_inherited': False, 'serving_qualification_inherited': False}
    if class_identity is None:
        return {'status': 'wait', 'reason': 'missing_actual_measurement'}
    low, remainder = divmod(rung * class_identity.get('arity', 0), 256)
    if not remainder or class_identity.get('run_widths') != [low, low + 1]:
        return {'status': 'wait', 'reason': 'different_geometry_class'}
    group = next((group for group in table.get('geometry_classes', []) if group['identity'] == class_identity), None)
    if group is None or len(group['observed_rungs']) < 2:
        return {'status': 'wait', 'reason': 'missing_measured_class_spots'}
    observed = group['observed_rungs']
    anchors = sorted(set((observed[0], observed[len(observed) // 2], observed[-1])))
    by_rung = {row['rung']: row for row in table['rungs']}
    cells = [next(cell for cell in by_rung[q]['measurements'] if cell['cell_id'] == selected_id) for q in anchors]
    times = [cell['kernel_time_us'] for cell in cells]
    return {'status': 'inherited', 'rung': rung, 'kernel_time_us': max(times),
            'observed_range_us': [min(times), max(times)], 'anchors': anchors,
            'action_keys': [cell['evidence'].get('action_key') for cell in cells], 'menu_admitted': False,
            'numerical_qualification_inherited': False, 'serving_qualification_inherited': False}



TABLE_SCHEMA_V3 = _copy.deepcopy(TABLE_SCHEMA_V2)
TABLE_SCHEMA_V3['$id'] = 'fleet.rung_allowability.v3'
TABLE_SCHEMA_V3['properties']['schema'] = {'const': 'fleet.rung_allowability.v3'}
TABLE_SCHEMA_V3['required'] += ['performant_policy', 'geometry_classes']
TABLE_SCHEMA_V3['properties']['performant_policy'] = {'type': 'object', 'required': ['kind'], 'properties': {'kind': {'const': 'whole_bit_per_structure'}}}
TABLE_SCHEMA_V3['properties']['geometry_classes'] = {'type': 'array', 'items': {'type': 'object', 'required': ['identity', 'observed_rungs'], 'properties': {'identity': {'type': 'object'}, 'observed_rungs': {'type': 'array', 'items': {'type': 'integer', 'minimum': 1}}}}}
TABLE_SCHEMAS['fleet.rung_allowability.v3'] = TABLE_SCHEMA_V3
_version_schema['properties']['table_schema']['enum'].append('fleet.rung_allowability.v3')



def rung_quality(rung, *, lower_rung, upper_rung, lower_value, upper_value):
    """Interpolate one scalar-bit interval; derived values are never measurements.

    Callers supply comparable actual unit/family/calibration anchors. Pair-grid
    overhead may shift integer-byte-rate anchors, so no q modulo is assumed.
    """
    _require(all(_integer(value) for value in (rung, lower_rung, upper_rung)) and
             upper_rung - lower_rung == 256 and lower_rung <= rung <= upper_rung, 'quality anchor interval')
    _require(_number(lower_value) and _number(upper_value), 'quality anchor value')
    fraction = (rung - lower_rung) / 256
    return {'status': 'derived', 'value': lower_value + fraction * (upper_value - lower_value),
            'anchors': [lower_rung, upper_rung], 'fraction': fraction,
            'numerical_qualification_inherited': False}

