"""CPU execution of the native paired-schedule admission and allocation owner.

These checks exercise actual CUDA-source pure C++ without a CUDA device. They
establish dispatch/allocation arithmetic, not GPU numerics or performance.
"""
import ast
from pathlib import Path
import os
import shutil
import subprocess

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/tessera/serving/csrc/routed_fused_window.cu'


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


def test_python_resource_helper_matches_native(native_owner):
    pytest.importorskip("torch")
    from tessera import routed_fused as rf
    for mode in (0,2):
        for paired in (False,True):
            assert rf.launch_smem_bytes(mode,8,mma8=True,bm=128,paired=paired) == _run(native_owner,mode,int(paired))
    for kwargs in ({'mma8':False},{'mode':1},{'bm':64},{'slot_words':4}):
        args={'mode':0,'slot_words':8,'mma8':True,'bm':128,'paired':True}
        args.update(kwargs)
        with pytest.raises(rf.GrammarError, match='paired shared memory requires'):
            rf.launch_smem_bytes(**args)


def test_compile_gate_is_scoped_and_frozen_at_import(monkeypatch):
    pytest.importorskip("torch")
    from tessera import routed_fused as rf
    flag = "-DTESSERA_ROUTED_FUSED_PAIRED_K32=1"
    monkeypatch.setattr(rf, "PAIRED_K32_BUILD", False)
    monkeypatch.setenv(rf.ENV_PAIRED_K32, "1")
    assert flag not in rf._cflags("sm_121", True, True)
    monkeypatch.setattr(rf, "PAIRED_K32_BUILD", True)
    monkeypatch.setenv(rf.ENV_PAIRED_K32, "0")
    assert flag in rf._cflags("sm_121", True, True)
    assert flag not in rf._cflags("sm_121", True, False)
    assert flag not in rf._cflags("sm_121", False, False, True)


@pytest.mark.parametrize("choice", [None, "0", "1", "", "true", "2", "-1"])
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
        with pytest.raises(GrammarError, match="compile flag must be 0 or 1"):
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
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
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
            // Previous epilogue is AFTER its last EMPTY, BEFORE next descriptor.
            epilogued=p.item-1;
        }
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
            // Prologue uses exactly one group and the ordinary one-step registers.
            issue_words(0,false);issue_words(1,false);cp_async_commit();
            int prev_nxt=0,cm_nxt=0,a_nxt=0;
            cur_k=0;const int nkc=2*np;
@PAIRED_BODY@
            require(pending.empty() && groups==0,"last group leaks into next item's LUT copies");
        }
        consume_to(gc-1);epilogued=items-1;
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
