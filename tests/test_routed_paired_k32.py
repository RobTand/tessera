"""CPU execution of the native paired-schedule admission and allocation owner.

These checks exercise actual CUDA-source pure C++ without a CUDA device. They
establish dispatch/allocation arithmetic, not GPU numerics or performance.
"""
import ast
import importlib.util
from pathlib import Path
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/tessera/serving/csrc/routed_fused_window.cu'


def _compile_cpp(root, name, source):
    cpp, binary = root / f'{name}.cpp', root / name
    cpp.write_text(source)
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    assert compiler, 'CPU source execution requires a C++ compiler'
    result = subprocess.run([compiler, '-std=c++17', '-O0', str(cpp), '-o', str(binary)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return binary


def _mutate(text, old, new):
    assert old in text, f'mutation target absent: {old}'
    return text.replace(old, new, 1)


@pytest.fixture
def fresh_paired(monkeypatch):
    pytest.importorskip('torch')

    def load(choice):
        monkeypatch.setenv('TESSERA_ROUTED_FUSED_PAIRED_K32', str(choice))
        monkeypatch.setenv('TESSERA_ROUTED_FUSED_MMA8_A_RING', '0')
        monkeypatch.setenv('TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH', '0')
        name = 'tessera._paired_k32_control'
        spec = importlib.util.spec_from_file_location(name, SOURCE.parents[2] / 'routed_fused.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return load


def _function_end(text, opening):
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return end


@pytest.fixture(scope='module')
def native_owner(tmp_path_factory):
    text = SOURCE.read_text()
    marker = '__host__ __device__ constexpr bool paired_k32_scope('
    assert marker in text, 'native paired scope owner is absent before this feature'
    end = _function_end(text, text.index('{', text.index(marker)))
    # The unchanged and paired layout share this authoritative constants owner.
    source = text[text.index('constexpr int THREADS ='):end]
    prefix = '''
#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#define __host__
#define __device__
#define TESSERA_ROUTED_FUSED_A_PREFETCH 4
constexpr bool FAMILY_FP8=true, FAMILY_MMA8=true, FAMILY_FP4=false;
constexpr bool PAIRED_K32_BUILD=true;
'''
    main = '''
int main(int argc, char** argv) {
    if (argc == 3) {
        int mode = std::atoi(argv[1]);
        bool paired = std::atoi(argv[2]);
        std::cout << launch_smem_bytes(mode, 8, BM_WIDE, paired); return 0;
    }
    if (argc != 12) return 2;
    std::cout << paired_k32_scope(std::atoi(argv[1]), std::atoi(argv[2]),
       std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
       std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]),
       std::atoi(argv[9]), std::atol(argv[10]), std::atoi(argv[11]));
}
'''
    root = tmp_path_factory.mktemp('paired-native-owner')
    cpp, binary = root/'owner.cpp', root/'owner'
    cpp.write_text(prefix + source + main)
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    assert compiler, 'CPU source-owner regression requires a C++ compiler'
    result = subprocess.run([compiler, '-std=c++17', '-O0', str(cpp), '-o', str(binary)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return binary


def _run(binary, *args):
    if len(args) == 10:
        args = (*args, 8)
    result = subprocess.run([str(binary), *map(str,args)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    return int(result.stdout)


@pytest.mark.parametrize('mode,original,paired', [(0,57552,76240),(2,40976,59664)])
def test_allocation_executes_native_owner(native_owner, mode, original, paired):
    assert _run(native_owner, mode, 0) == original
    assert _run(native_owner, mode, 1) == paired


@pytest.mark.parametrize('mode', [0,2])
@pytest.mark.parametrize('K,rows', [(192,512),(256,2048),(4096,2049)])
def test_admitted_prefill(native_owner, mode, K, rows):
    assert _run(native_owner,1,1,mode,0,0,4,0,128,K,rows) == 1


@pytest.mark.parametrize('field,value', [
    (0,0),(1,0),(2,1),(3,1),(4,1),(5,3),(5,5),(6,1),(7,64),
    (8,128),(8,160),(8,224),(10,12),(10,16),(9,1),(9,2),(9,4),(9,8),(9,511),
])
def test_unsupported_call_uses_original(native_owner, field, value):
    args = [1,1,0,0,0,4,0,128,192,512,8]
    args[field] = value
    assert _run(native_owner,*args) == 0


def test_python_resource_helper_matches_native(native_owner, fresh_paired):
    rf = fresh_paired(0)
    for mode in (0,2):
        for paired in (False,True):
            assert rf.launch_smem_bytes(mode,8,mma8=True,bm=128,paired=paired) == _run(native_owner,mode,int(paired))
    for kwargs in ({'mma8':False},{'mode':1},{'bm':64},{'slot_words':4}):
        args={'mode':0,'slot_words':8,'mma8':True,'bm':128,'paired':True}
        args.update(kwargs)
        with pytest.raises(rf.GrammarError, match='paired shared memory requires'):
            rf.launch_smem_bytes(**args)


def test_compile_gate_is_scoped_and_frozen_at_import(monkeypatch, fresh_paired):
    flag = "-DTESSERA_ROUTED_FUSED_PAIRED_K32=1"
    rf = fresh_paired(0)
    monkeypatch.setenv(rf.ENV_PAIRED_K32, "1")
    assert flag not in rf._cflags("sm_121", True, True)
    rf = fresh_paired(1)
    monkeypatch.setenv(rf.ENV_PAIRED_K32, "0")
    assert flag in rf._cflags("sm_121", True, True)
    assert flag not in rf._cflags("sm_121", True, False)
    assert flag not in rf._cflags("sm_121", False, False, True)


@pytest.mark.parametrize("choice", [None, "0", "1", "", "true", "2", "-1", "01", " 1", "1\n"])
def test_paired_import_choice_is_strict_and_defaults_off(choice):
    from types import SimpleNamespace
    from tessera.errors import GrammarError
    path = SOURCE.parents[2] / "routed_fused.py"
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        if "_paired_k32_choice" in names or (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "ENV_PAIRED_K32" for target in node.targets)):
            selected.append(node)
    env = {} if choice is None else {"TESSERA_ROUTED_FUSED_PAIRED_K32": choice}
    scope = {"os": SimpleNamespace(environ=env), "GrammarError": GrammarError}
    program = compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec")
    if choice not in (None, "0", "1"):
        with pytest.raises(GrammarError, match="TESSERA_ROUTED_FUSED_PAIRED_K32=.*compile flag must be 0 or 1"):
            exec(program, scope)
    else:
        exec(program, scope)
        assert scope["PAIRED_K32_BUILD"] is (choice == "1")


PRODUCER_HARNESS = r'''
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
constexpr int THREADS=512, PRODUCER_THREADS=256, BAR_PROD=5, BAR_EMPTY0=3, BAR_FULL0=1;
constexpr bool PREFETCH_A=true;
constexpr int A_PREFETCH=4;
struct Copy {int item,kc;};
std::vector<Copy> pending;
std::array<Copy,4> words;
std::array<bool,4> read_words{};
struct Pair {int item,kc; bool published=false;};
std::vector<Pair> published;
unsigned gc=0, consumer=0;
int groups=0;
int cur_item=0, cur_k=-1, next_k=-1, np=0, epilogued=-1;
int wscale[2]={-1,-1};
int descriptor[2]={-1,-1};
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
void finish_epilogue(int item) {
    if(item<0) return;
    require(descriptor[item&1]==item,"descriptor slot overwritten before delayed epilogue");
    require(wscale[item&1]==item,"scale slot overwritten before delayed epilogue");
    epilogued=item;
}
void issue_words(int kc,bool ring) {
    require(!ring,"one-run schedule borrowed descriptor ring");
    int slot=((kc/2)&1)*2+(kc&1);
    require(words[slot].item<0 || read_words[slot],"word slot overwritten before final producer reader");
    pending.push_back({cur_item,kc});
}
void cp_async_commit() {++groups;}
template<int N> void cp_async_wait() {
    require(N==0,"paired schedule did not wait for sole current group");
    for(auto c:pending) {int slot=((c.kc/2)&1)*2+(c.kc&1);words[slot]=c;read_words[slot]=false;}
    pending.clear();groups=0;
}
void consume_to(unsigned target) {
    while(consumer<=target) {
        require(consumer<published.size() && published[consumer].published,"consumer before FULL");
        const Pair p=published[consumer];
        if(p.kc==0) {
            finish_epilogue(p.item-1);
        }
        require(descriptor[p.item&1]==p.item,"descriptor slot overwritten before consumer read");
        require(wscale[p.item&1]==p.item,"descriptor/scale slot overwritten before consumer read");
        ++consumer; // last EMPTY leaves its epilogue deliberately delayed
    }
}
void bar_sync(int id,int count) {
    if(id==BAR_PROD) {require(count==PRODUCER_THREADS,"producer count");return;}
    require(count==THREADS && id==BAR_EMPTY0+(gc&1),"EMPTY identity/count");
    require(gc>=2,"wait on first-use parity");consume_to(gc-2);
}
void bar_arrive(int id,int count) {
    require(count==THREADS && id==BAR_FULL0+(gc&1),"FULL identity/count");
    require(gc<published.size() && !published[gc].published,"FULL phase overwritten");
    published[gc].published=true;
}
void prefetch_a(int kc) {require(kc>=0 && kc<2*np,"prefetch beyond logical K");}
int main(int argc,char** argv) {
    np=std::atoi(argv[1]);int items=std::atoi(argv[2]);
    for(auto& w:words) w={-1,-1};
    auto load_prev=[&](int kc, auto&,auto&,bool ring) {require(!ring,"paired mapped from ring"); next_k=kc;};
    auto load_a=[&](int kc, auto&) {require(next_k==kc,"activation/map lookahead differ");};
    auto advance_micro=[&]() {cur_k=next_k;};
    auto publish_micro=[&](int kc,int micro) {
        require(cur_k==kc,"register pipeline changed K32 order");
        require(gc<2 || consumer>gc-2,"decoded stage overwritten before consumer EMPTY");
        require(micro==2*(gc&1)+(kc&1),"decoded microstage parity differs");
        int slot=((kc/2)&1)*2+(kc&1);
        require(words[slot].item==cur_item && words[slot].kc==kc,"decode before word/history group completed");
        read_words[slot]=true;
        if(!(kc&1)) published.push_back({cur_item,kc,false});
        else require(published[gc].kc+1==kc,"MMA pair order changed");
    };
    try {
        for(cur_item=0;cur_item<items;++cur_item) {
            require(cur_item<2 || epilogued>=cur_item-2,"item slot overwrite raced last epilogue");
            wscale[cur_item&1]=cur_item;
            descriptor[cur_item&1]=cur_item;
            // Prologue uses exactly one group and the ordinary one-step registers.
            issue_words(0,false);issue_words(1,false);cp_async_commit();
            int prev_nxt=0,cm_nxt=0,a_nxt=0;
            cur_k=0;const int nkc=2*np;
@PAIRED_BODY@
            require(pending.empty() && groups==0,"last group leaks into next item's LUT copies");
        }
        consume_to(gc-1);finish_epilogue(items-1);
        std::cout<<"ordered K32 pairs="<<gc<<" items="<<items;
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what();return 1;}
}
'''


@pytest.fixture(scope='module')
def native_producer(tmp_path_factory):
    text=SOURCE.read_text()
    marker='const int np = nkc / 2;'
    assert marker in text, 'paired producer is absent before this feature'
    start=text.rfind('if constexpr (PAIRED) {',0,text.index(marker))
    opening=text.index('{',start)
    body=text[opening+1:_function_end(text,opening)-1]
    root=tmp_path_factory.mktemp('paired-producer')
    cpp,binary=root/'producer.cpp',root/'producer'
    cpp.write_text(PRODUCER_HARNESS.replace('@PAIRED_BODY@',body))
    compiler=shutil.which(os.environ.get('CXX','c++'))
    assert compiler
    result=subprocess.run([compiler,'-std=c++17','-O0',str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return binary


@pytest.mark.parametrize('pairs,items',[(3,3),(3,5),(4,3),(64,5)])
def test_native_producer_lifetimes_and_logical_order(native_producer,pairs,items):
    result=subprocess.run([str(native_producer),str(pairs),str(items)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr


def test_two_pair_item_is_causal_scale_lifetime_control(native_producer):
    result=subprocess.run([str(native_producer),'2','3'],capture_output=True,text=True,timeout=5)
    assert result.returncode==1
    assert 'item slot overwrite raced last epilogue' in result.stderr


@pytest.mark.parametrize('mutation,expected', [
    ('wait', 'decode before word/history group completed'),
    ('empty', 'decoded stage overwritten before consumer EMPTY'),
    ('order', 'register pipeline changed K32 order'),
    ('drain', "last group leaks into next item's LUT copies"),
])
def test_producer_model_catches_unsafe_source_mutants(tmp_path, mutation, expected):
    text=SOURCE.read_text()
    marker='const int np = nkc / 2;'
    start=text.rfind('if constexpr (PAIRED) {',0,text.index(marker))
    opening=text.index('{',start)
    body=text[opening+1:_function_end(text,opening)-1]
    if mutation=='wait':
        body=body.replace('cp_async_wait<0>();','/* causal missing copy wait */')
    elif mutation=='empty':
        body=body.replace('if (gc >= 2) bar_sync(BAR_EMPTY0 + stage, THREADS);', '/* causal missing EMPTY */')
    elif mutation=='order':
        body=body.replace('publish_micro(kc + 1, 2 * stage + 1);','publish_micro(kc, 2 * stage + 1);')
    else:
        position=body.rfind('cp_async_wait<0>();')
        body=body[:position]+body[position:].replace('cp_async_wait<0>();','/* causal missing final drain */',1)
    cpp,binary=tmp_path/'mutant.cpp',tmp_path/'mutant'
    cpp.write_text(PRODUCER_HARNESS.replace('@PAIRED_BODY@',body))
    compiler=shutil.which(os.environ.get('CXX','c++'))
    assert compiler
    result=subprocess.run([compiler,'-std=c++17','-O0',str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(binary),'3','3'],capture_output=True,text=True,timeout=5)
    assert result.returncode==1
    assert expected in result.stderr


LAUNCH_CASES = [
    (0, False, False, 4, False, 128, False),
    (2, False, False, 4, False, 128, False),
    (1, False, False, 4, False, 128, False),
    (2, True, False, 4, False, 128, False),
    (2, True, True, 4, False, 64, False),
    (0, False, False, 3, False, 128, False),
    (0, False, False, 5, False, 128, False),
    (0, False, False, 4, True, 128, False),
    (0, False, False, 4, False, 64, False),
    (0, False, False, 4, False, 128, True),
    (2, False, False, 4, False, 128, True),
]


LAUNCH_HARNESS = r'''
#include <sstream>
#include <stdexcept>
#include <array>
using cudaStream_t = int;
struct Params { int K, slot_words, tile_words; long rows_x; bool piece_major; };
namespace torch {
struct Device { int index() const { return 7; } };
struct Tensor {
    long rows;
    long size(int axis) const { if (axis != 0) throw std::runtime_error("unexpected axis"); return rows; }
    Device device() const { return {}; }
};
}
int capacity, probed_device = -1;
int max_dynamic_smem_bytes(int device) { probed_device = device; return capacity; }
template<class... Args> void check(bool ok, Args... args) {
    if (!ok) { std::ostringstream out; (out << ... << args); throw std::runtime_error(out.str()); }
}
#define TORCH_CHECK(ok, ...) check(ok, __VA_ARGS__)
@CHECK_SLOT@
bool launched_pair = false;
int allocation = -1;
std::array<int,7> launched_geometry;
template <bool FP8, int MODE, bool DENSE, bool SPLIT, int RL, bool TWO, int BMT, bool PM, bool PAIRED = false>
void launch_variant(const Params& p, int grid, cudaStream_t stream) {
    check(grid == 3 && stream == 11, "launch arguments changed");
    check(FP8 == FAMILY_FP8, "launch family changed");
    launched_pair = PAIRED;
    launched_geometry = {MODE,DENSE,SPLIT,RL,TWO,BMT,PM};
    allocation = launch_smem_bytes(MODE, p.slot_words, BMT, PAIRED);
}
@DISPATCH@
template<int MODE, bool DENSE, bool SPLIT, int RL, bool TWO, int BMT, bool PM>
void run_case(int K, long rows, int slot_words) {
    const int mode = MODE, bm = BMT;
    constexpr bool FP8 = FAMILY_FP8;
    const bool piece_major = PM;
    const int tile_words = 16 * (K * RL + (TWO ? K / 2 : 0));
    torch::Tensor x{rows};
    Params p{K,slot_words,tile_words,rows,piece_major};
@WIDTH@
    bool admitted = false;
    if constexpr (!DENSE && !SPLIT) {
@ENTRY@
        admitted = paired;
    } else {
        check_slot(mode, slot_words, x, bm);
    }
    check(probed_device == 7, "check_slot did not query the tensor's device");
@PIECE_MAJOR_FAMILY@
    launch_pair<FAMILY_FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM>(p, 3, 11);
    check(launched_geometry == std::array<int,7>{MODE,DENSE,SPLIT,RL,TWO,BMT,PM},
          "fallback specialization changed");
    std::cout << admitted << ' ' << launched_pair << ' ' << allocation;
}
int main(int argc, char** argv) {
    const int which = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]), rows = std::atoi(argv[3]), slot_words = std::atoi(argv[4]);
    capacity = std::atoi(argv[5]);
    try {
        switch(which) {
@CASES@
        default: throw std::runtime_error("unknown case");
        }
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what(); return 1; }
}
'''


def _native_owners(text, enabled, fp8, mma8):
    end = _function_end(text, text.index('{', text.index('constexpr PairKey pair_of(')))
    owners = text[text.index('constexpr int THREADS ='):end]
    build_start = text.index('constexpr bool PAIRED_K32_BUILD =')
    build = text[build_start:text.index(';', build_start) + 1]
    prefix = f'''
#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#define __host__
#define __device__
#define TESSERA_ROUTED_FUSED_PAIRED_K32 {int(enabled)}
constexpr bool FAMILY_FP8={str(fp8).lower()}, FAMILY_MMA8={str(mma8).lower()}, FAMILY_FP4=false;
{build}
'''
    return prefix + owners


def _native_launch_source(text, enabled, fp8, mma8):
    dispatch_start = text.index('template <bool FP8, int MODE, bool DENSE, bool SPLIT, int RL, bool TWO, int BMT, bool PM = false>\nvoid launch_pair(')
    dispatch_end = _function_end(text, text.index('{', dispatch_start))
    slot_start = text.index('void check_slot(')
    slot_end = _function_end(text, text.index('{', slot_start))
    entry_start = text.index('const PairKey schedule_pair =')
    entry_end = text.index('check_slot((int)mode, slot_words, x, (int)bm, paired);', entry_start)
    entry_end += len('check_slot((int)mode, slot_words, x, (int)bm, paired);')
    cases = '\n'.join(
        f'case {i}: run_case<{",".join(str(int(v)) for v in geometry)}>(K,rows,slot_words); break;'
        for i, geometry in enumerate(LAUNCH_CASES))
    width_start = text.index('TORCH_CHECK(has_width(FAMILY_FP8, FAMILY_MMA8, (int)mode, (int)bm)')
    width_end = text.index(');', width_start) + 2
    pm_start = text.index('TORCH_CHECK(!p.piece_major || (FAMILY_MMA8 && FP8)')
    pm_end = text.index(');', pm_start) + 2
    harness = (LAUNCH_HARNESS.replace('@CHECK_SLOT@', text[slot_start:slot_end])
               .replace('@DISPATCH@', text[dispatch_start:dispatch_end])
               .replace('@ENTRY@', text[entry_start:entry_end])
               .replace('@WIDTH@', text[width_start:width_end])
               .replace('@PIECE_MAJOR_FAMILY@', text[pm_start:pm_end])
               .replace('@CASES@', cases))
    return _native_owners(text, enabled, fp8, mma8) + harness


@pytest.fixture(scope='module')
def native_launch(tmp_path_factory):
    root = tmp_path_factory.mktemp('paired-launch')
    binaries = {}

    def build(enabled=True, fp8=True, mma8=True, mutation=None):
        key = enabled, fp8, mma8, mutation
        if key not in binaries:
            text = SOURCE.read_text()
            if mutation == 'disabled':
                text = _mutate(text, 'return PAIRED_K32_BUILD && fp8 && mma8', 'return fp8 && mma8')
            elif mutation == 'piece-major-entry':
                text = _mutate(text, 'const bool paired = !piece_major', 'const bool paired = true')
            elif mutation == 'piece-major-dispatch':
                text = _mutate(text, 'if constexpr (!PM && PAIRED_K32_BUILD', 'if constexpr (PAIRED_K32_BUILD')
            elif mutation == 'allocation':
                text = _mutate(text, 'launch_smem_bytes(mode, (int)slot_words, bmt, paired);',
                               'launch_smem_bytes(mode, (int)slot_words, bmt, false);')
            binaries[key] = _compile_cpp(root, f'launch{len(binaries)}',
                                        _native_launch_source(text, enabled, fp8, mma8))
        return binaries[key]

    return build


def _launch(binary, case, K=192, rows=512, words=8, capacity=101376):
    return subprocess.run([str(binary), *map(str, (case, K, rows, words, capacity))],
                          text=True, capture_output=True, timeout=5)


def _assert_selection(result, paired):
    assert result.returncode == 0, result.stderr
    entry, dispatch, allocation = map(int, result.stdout.split())
    assert entry == dispatch == int(paired), f'entry={entry}, dispatch={dispatch}, expected={int(paired)}'
    return allocation


def test_disabled_native_build_never_admits_pairing(native_launch):
    binary = native_launch(enabled=False)
    for case in range(len(LAUNCH_CASES)):
        _assert_selection(_launch(binary, case), False)
    with pytest.raises(AssertionError, match='entry=1, dispatch=0') as failure:
        _assert_selection(_launch(native_launch(enabled=False, mutation='disabled'), 0), False)
    print(f'unsafe disabled-build mutation: {failure.value}')


@pytest.mark.parametrize('enabled,fp8,mma8', [(True, True, True), (False, True, True),
                                           (True, True, False), (True, False, False)])
def test_paired_entry_and_dispatch_agree(native_launch, enabled, fp8, mma8):
    binary = native_launch(enabled, fp8, mma8)
    for case in range(len(LAUNCH_CASES)):
        for K, rows, words in [(192, 512, 8), (256, 2049, 8), (128, 512, 8),
                               (160, 512, 8), (224, 512, 8), (192, 511, 8),
                               (192, 512, 12), (192, 512, 16)]:
            expected = (enabled and fp8 and mma8 and case in (0, 1)
                        and (K, rows, words) in ((192, 512, 8), (256, 2049, 8)))
            result = _launch(binary, case, K, rows, words)
            mode, _, _, _, _, bmt, pm = LAUNCH_CASES[case]
            if not mma8 and bmt == 128 and (not fp8 or mode != 2):
                assert result.returncode == 1
                assert 'has no' in result.stderr
            elif pm and not mma8:
                assert result.returncode == 1
                assert 'piece-major reader is the E4M3 MMA reader only' in result.stderr
            else:
                _assert_selection(result, expected)
    if enabled and fp8 and mma8:
        for mutation, diagnostic in [('piece-major-entry', 'entry=1, dispatch=0'),
                                     ('piece-major-dispatch', 'entry=0, dispatch=1')]:
            with pytest.raises(AssertionError, match=diagnostic) as failure:
                _assert_selection(_launch(native_launch(mutation=mutation), 9), False)
            print(f'unsafe {mutation} mutation: {failure.value}')


@pytest.mark.parametrize('case,original,paired', [(0, 57552, 76240), (1, 40976, 59664)])
def test_paired_live_device_guard_checks_launched_allocation(native_launch, case, original, paired):
    binary = native_launch()
    assert _assert_selection(_launch(binary, case, capacity=paired), True) == paired
    refused = _launch(binary, case, capacity=paired - 1)
    assert refused.returncode == 1, refused.stdout
    assert f'needs {paired} bytes' in refused.stderr
    assert f'device 7 allows {paired - 1}' in refused.stderr
    for K, rows in [(128, 512), (192, 511)]:
        assert _assert_selection(_launch(binary, case, K, rows, capacity=original), False) == original
        assert _launch(binary, case, K, rows, capacity=original - 1).returncode == 1
    assert _assert_selection(_launch(native_launch(enabled=False), case, capacity=original), False) == original
    pm_case = 9 if case == 0 else 10
    assert _assert_selection(_launch(binary, pm_case, capacity=original), False) == original
    mutant = _launch(native_launch(mutation='allocation'), case, capacity=paired - 1)
    with pytest.raises(AssertionError, match='guard admitted insufficient capacity') as failure:
        assert mutant.returncode == 1, f'guard admitted insufficient capacity: {mutant.stdout}'
    print(f'unsafe live-allocation mutation: {failure.value}')


CONSUMER_HARNESS = r'''
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>
using std::min;
using std::max;
constexpr int BAR_FULL0=1, BAR_EMPTY0=3;
constexpr int BMT=128, WROWS=BMT/2, MI=WROWS/16, A_STAGE=BMT*BK;
constexpr bool FP8=true, DENSE=false, SPLIT=false, TWO=false;
constexpr int RL=4;
using Traces = std::array<std::vector<int>,MI*4*4>;
struct Context {
    int micros, nkc, mb, mw, q=-1, full=0, empty=0, b_reads=0;
    bool released=true;
    uintptr_t acc_base=0;
    const uint8_t *As, *Bs;
    Traces traces;
} *context;
void require(bool ok, const char* message) { if(!ok) throw std::runtime_error(message); }
void bar_sync(int id, int count) {
    auto& c=*context;
    require(count==THREADS && id==BAR_FULL0+(c.full&1),"FULL parity/count changed");
    require(c.released,"next FULL overwrote an unacknowledged decoded pair");
    c.q=c.full++;
    c.released=false;
}
void bar_arrive(int id, int count) {
    auto& c=*context;
    require(count==THREADS && id==BAR_EMPTY0+(c.q&1),"EMPTY parity/count changed");
    require(!c.released,"more than one EMPTY per quantum");
    require(c.b_reads==(c.q+1)*c.micros*2,"EMPTY before final accumulator reader");
    for(int mi=0;mi<MI;++mi) for(int nt=0;nt<4;++nt) for(int i=0;i<4;++i) {
        const int expected=(64*c.mw+16*mi<c.mb) ? (c.q+1)*c.micros : 0;
        require(c.traces[mi*16+nt*4+i].size()==unsigned(expected),
                "EMPTY before final accumulator reader");
    }
    c.released=true;
    ++c.empty;
}
int fragment_k(const void* address, bool activation) {
    auto& c=*context;
    require(!c.released,"fragment read after EMPTY");
    const auto* base=activation ? c.As : c.Bs;
    const int stride=activation ? A_STAGE : B_STAGE_BYTES;
    const auto offset=reinterpret_cast<uintptr_t>(address)-reinterpret_cast<uintptr_t>(base);
    require(offset<unsigned(2*c.micros*stride),"fragment outside decoded pair slots");
    const int micro=offset/stride;
    require(micro/c.micros==(c.q&1),"fragment read wrong pair parity");
    return c.q*c.micros+micro%c.micros;
}
void ldmatrix_x4(uint32_t (&a)[4], const void* address) {
    const int k=fragment_k(address,true);
    for(auto& word:a) word=k+1;
}
void ldmatrix_x4_trans(uint32_t (&b)[4], const void* address) {
    const int k=fragment_k(address,false);
    ++context->b_reads;
    for(auto& word:b) word=k+1;
}
uint32_t __byte_perm(uint32_t x,uint32_t y,int selector) {
    require(x==y && (selector==0x6420 || selector==0x7531),"B fragment shim inputs changed");
    return x;
}
void mma16832_e4m3(float (&acc)[4], uint32_t (&a)[4], uint32_t (&b)[2]) {
    auto& c=*context;
    require(!c.released,"MMA read after EMPTY");
    for(auto word:a) require(word==b[0] && word==b[1],"MMA operands refer to different K32 chunks");
    const int index=(reinterpret_cast<uintptr_t>(&acc[0])-c.acc_base)/sizeof(float);
    require(index>=0 && index+3<MI*4*4,"unknown accumulator");
    for(int i=0;i<4;++i) c.traces[index+i].push_back(int(a[0])-1);
}
int b8swz(int) { return 0; }
int aswz(int,int) { return 0; }
int bswz(int,int) { return 0; }
template<bool> void mma16816(float (&)[4],uint32_t (&)[4],uint32_t (&)[2]) {
    throw std::runtime_error("unexpected f16 instruction");
}
template<bool PAIRED,int MODE>
Traces consume(int nkc,int mb,int cw,int lane) {
    using L=Layout<MODE,BMT,PAIRED>;
    std::array<uint8_t,4*A_STAGE> activation{};
    std::array<uint8_t,4*B_STAGE_BYTES> weights{};
    const auto* As=activation.data();
    const auto* Bs=weights.data();
    const int mw=cw>>2, nw=cw&3;
    unsigned gc=0;
    int stage=0;
    Context c{L::MICROS,nkc,mb,mw};
    c.As=As; c.Bs=Bs;
    context=&c;
    bar_sync(BAR_FULL0,THREADS);
@ACCUMULATORS@
    c.acc_base=reinterpret_cast<uintptr_t>(&acc[0][0][0]);
@LOOP@
    require(c.released && c.full==nkc/L::MICROS && c.empty==c.full,
            "consumer quantum handoff count changed");
    require(gc==unsigned(c.full),"consumer quantum counter changed");
    for(int mi=0;mi<MI;++mi) for(int nt=0;nt<4;++nt) for(int i=0;i<4;++i) {
        auto& trace=c.traces[mi*16+nt*4+i];
        const bool live=64*mw+16*mi<mb;
        require(trace.size()==unsigned(live ? nkc : 0),"live/padding accumulator coverage changed");
        for(int k=0;k<int(trace.size());++k)
            require(trace[k]==k,"accumulator K32 order changed");
    }
    return c.traces;
}
template<int MODE> void check_mode(int nkc) {
    for(int mb: {0,1,15,16,17,63,64,65,79,80,81,127,128})
        for(int cw=0;cw<8;++cw) for(int lane=0;lane<32;++lane)
            require(consume<true,MODE>(nkc,mb,cw,lane)==consume<false,MODE>(nkc,mb,cw,lane),
                    "paired/original accumulator traces differ");
}
int main(int argc,char** argv) {
    try {
        int nkc=std::atoi(argv[1]);
        if(std::atoi(argv[2])==0) check_mode<0>(nkc); else check_mode<2>(nkc);
        std::cout<<"per-accumulator traces agree; K32 chunks="<<nkc;
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what(); return 1; }
}
'''


@pytest.fixture(scope='module')
def native_consumer(tmp_path_factory):
    root = tmp_path_factory.mktemp('paired-consumer')
    binaries = {}

    def build(mutation=None):
        if mutation not in binaries:
            text = SOURCE.read_text()
            start = text.index('const int live = (BMT == BM) ? MI')
            loop_start = text.index('for (int ic = 0; ic < nkc / L::MICROS;', start)
            loop_end = _function_end(text, text.index('{', loop_start))
            accumulators, loop = text[start:loop_start], text[loop_start:loop_end]
            if mutation == 'order':
                loop = _mutate(loop, 'stage * L::MICROS + u',
                               'stage * L::MICROS + (L::MICROS - 1 - u)')
            elif mutation == 'early-empty':
                loop = _mutate(loop, 'const int micro =',
                               'if (u == 1) bar_arrive(BAR_EMPTY0 + stage, THREADS);\nconst int micro =')
            harness = (CONSUMER_HARNESS.replace('@ACCUMULATORS@', accumulators)
                       .replace('@LOOP@', loop))
            owner = _native_owners(text, True, True, True)
            binaries[mutation] = _compile_cpp(root, f'consumer{len(binaries)}', owner + harness)
        return binaries[mutation]

    return build


def _consume(binary, chunks=6, mode=0):
    return subprocess.run([str(binary), str(chunks), str(mode)], text=True,
                          capture_output=True, timeout=30)


@pytest.mark.parametrize('chunks', [6, 8, 128])
@pytest.mark.parametrize('mode', [0, 2])
def test_paired_consumer_preserves_each_accumulator_k32_order(native_consumer, chunks, mode):
    result = _consume(native_consumer(), chunks, mode)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mutation,diagnostic', [('order', 'accumulator K32 order changed'),
                                                ('early-empty', 'EMPTY before final accumulator reader')])
def test_paired_consumer_proof_rejects_order_and_early_empty_mutants(native_consumer, mutation, diagnostic):
    result = _consume(native_consumer(mutation))
    with pytest.raises(AssertionError, match=diagnostic) as failure:
        assert result.returncode == 0, result.stderr
    print(f'unsafe consumer {mutation} mutation: {failure.value}')


def _compiled_metadata(paired):
    """A compiled-metadata fixture at default independent MMA8 controls."""
    return SimpleNamespace(
        BM=64, BN=128, HALF=64, BK=32, DENSE_ROW_QUANTUM=4,
        RATE_MIN=1, ROUTED_RATE_MAX=8, RATE_MAX=8, SLOT_WORDS_MAX=16,
        BDESC_INTS=12, WINDOW_BITS=14, FAMILY_FP8=True, FAMILY_MMA8=True,
        PAIRED_K32_BUILD=paired, WORD_STAGES=3, WORD_STAGES_MIN=2,
        STAGES=2, MAX_ROLES=8, SMEM_FIXED_GATE_UP=47312, SMEM_FIXED_DOWN=30736,
        BM_WIDE=128, A_REGION_BYTES_WIDE=8192, HAS_WIDE_GATE_UP=True,
        HAS_WIDE_DOWN=True, GATE_UP_RATE_MAX=8, MMA8_GATE_UP_B_PREFETCH=False,
        MMA8_A_RING=0)


@pytest.mark.parametrize('selected,compiled', [(0, True), (1, False)])
def test_paired_compiled_selector_mismatch_is_refused(fresh_paired, monkeypatch, selected, compiled):
    rf = fresh_paired(selected)
    lib = _compiled_metadata(bool(selected))
    monkeypatch.setattr(rf, 'build_library', lambda *args: lib)
    assert rf._ext('e4m3mma') is lib
    rf._ext.cache_clear()
    lib.PAIRED_K32_BUILD = compiled
    with pytest.raises(rf.GrammarError, match='PAIRED_K32_BUILD'):
        rf._ext('e4m3mma')

    tree = ast.parse(Path(rf.__file__).read_text())
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_ext')
    matches = [node for node in ast.walk(loader) if isinstance(node, ast.Tuple)
               and len(node.elts) == 2 and isinstance(node.elts[0], ast.Constant)
               and node.elts[0].value == 'PAIRED_K32_BUILD']
    assert len(matches) == 1
    matches[0].elts[1] = ast.Constant(value=compiled)
    loader.decorator_list = []
    mutant = ast.fix_missing_locations(ast.Module(body=[loader], type_ignores=[]))
    namespace = dict(vars(rf))
    exec(compile(mutant, rf.__file__, 'exec'), namespace)
    with pytest.raises(pytest.fail.Exception, match='DID NOT RAISE') as failure:
        with pytest.raises(rf.GrammarError, match='PAIRED_K32_BUILD'):
            namespace['_ext']('e4m3mma')
    print(f'unsafe compiled-selector validation mutation: {failure.value}')
