ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PATCH_FILES
ARG KERNEL_BUILD
COPY images/serving/patches/ /opt/tessera-image/patches/
COPY images/serving/apply_patches.py /opt/tessera-image/apply_patches.py
RUN python3 /opt/tessera-image/apply_patches.py --patch-files "${PATCH_FILES}"
COPY pyproject.toml MANIFEST.in README.md LICENSE /opt/tessera-source/
COPY src/ /opt/tessera-source/src/
RUN python3 -m pip install --no-deps --no-build-isolation /opt/tessera-source
LABEL io.tessera.kernel-build=${KERNEL_BUILD}
ENV TESSERA_KERNEL_BUILD=${KERNEL_BUILD}
