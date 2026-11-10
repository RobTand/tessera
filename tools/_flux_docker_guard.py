"""Refuse legacy root Docker entry points inside a Flux job."""
import os
import sys

REASON = ('ROOT_DOCKER_REQUIRES_REVIEWED_SERVING_WRAPPER: root Docker in a Flux job '
          'only through the reviewed serving wrapper; anything else goes to the lead')


def refuse_unreviewed_docker():
    if 'FLUX_JOB_ID' in os.environ:
        raise SystemExit('REFUSED: ' + REASON)


if __name__ == '__main__':
    if 'FLUX_JOB_ID' in os.environ:
        print('REFUSED: ' + REASON, file=sys.stderr)
        sys.exit(2)
