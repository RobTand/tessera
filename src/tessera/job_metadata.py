"""Job identity at the explicit metadata schema boundary.

Unversioned historical records carry pb_action; v2 records carry flux_job_id.
A bare flux_job_id does not turn an unversioned record into v2.
"""
JOB_META_SCHEMA = 'tessera.job_meta.v2'
GEOMETRY_META_SCHEMA = 'tessera.geometry_meta.v2'


def job_identity(meta):
    schema = meta.get('meta_schema')
    if schema in (JOB_META_SCHEMA, GEOMETRY_META_SCHEMA):
        return meta.get('flux_job_id')
    if schema is None:
        return meta.get('pb_action')
    raise ValueError(f'unknown job meta schema: {schema}')
