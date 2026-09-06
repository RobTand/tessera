#!/usr/bin/env bash
# #375 probe: is a container's id in hand before its start can fail, and does
# PrismaBuild's docker shim accept the spelling that gets it?
#
# The obvious spelling of ownership is `docker create` (which prints the id)
# followed by `docker start`.  Section A measures what PB's shim does with the
# second half of that; sections B and C measure the `--cidfile` spelling the
# helper uses instead, including the case that leaks: a container created and
# then failed to start.
#
# CPU only, no GPU, no published ports, containers named tess375-cid-* and
# removed here.  Run it through PrismaBuild, not by hand.
set -u
D=$(mktemp -d "${TMPDIR:-/home/rob/tmp}/cidprobe.XXXXXX") || exit 1
echo "== which docker: $(command -v docker)"
echo
echo "== A) docker start is refused by the PB shim (this is why create+start is out)"
docker create --name tess375-cid-a ubuntu:24.04 sleep 5; echo "create rc=$?"
docker start tess375-cid-a; echo "start rc=$?"
docker rm -f tess375-cid-a >/dev/null 2>&1; echo "rm rc=$?"
echo
echo "== B) run -d --cidfile, healthy container"
docker run -d --cidfile "$D/b.cid" --name tess375-cid-b ubuntu:24.04 sleep 3 >/dev/null 2>&1
echo "run rc=$?  cidfile=[$(cat "$D/b.cid" 2>/dev/null)]"
docker rm -f "$(cat "$D/b.cid" 2>/dev/null)" >/dev/null 2>&1; echo "rm-by-id rc=$?"
echo
echo "== C) run -d --cidfile, start FAILS (bad entrypoint)"
docker run -d --cidfile "$D/c.cid" --name tess375-cid-c --entrypoint /no/such/binary ubuntu:24.04
echo "run rc=$?"
echo "cidfile=[$(cat "$D/c.cid" 2>/dev/null)]"
echo "container still present by id? [$(docker inspect -f '{{.State.Status}}' "$(cat "$D/c.cid" 2>/dev/null)" 2>/dev/null)]"
docker rm -f "$(cat "$D/c.cid" 2>/dev/null)" >/dev/null 2>&1; echo "rm-by-id rc=$?"
docker rm -f tess375-cid-c >/dev/null 2>&1
rm -rf "$D"
