#!/usr/bin/env bash
for f in /mnt/shared/models/GLM-5.3-Flash-4layer/model-*.safetensors; do dd if="$f" iflag=nocache count=0 status=none; done
awk '/^Cached/{print "Cached MiB after evict:", int($2/1024)}' /proc/meminfo
