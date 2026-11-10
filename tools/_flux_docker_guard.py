"""Refuse legacy root Docker entry points inside a Flux job."""
import os
import sys

REASON = ('ROOT_DOCKER_REQUIRES_REVIEWED_SERVING_WRAPPER: root Docker in a Flux job '
          'only through the reviewed serving wrapper; anything else goes to the lead')


def refuse_unreviewed_docker():
    if os.environ.get('FLUX_JOB_ID'):
        raise SystemExit('REFUSED: ' + REASON)


if __name__ == '__main__':
    if os.environ.get('FLUX_JOB_ID'):
        print('REFUSED: ' + REASON, file=sys.stderr)
        sys.exit(2)
