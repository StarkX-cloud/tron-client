def __call__(self, *args, **kwargs):

    dep_ids = [d.job_id for d in self.depends_on if hasattr(d, "job_id")]

    job_id = submit(
        self.fn,
        gpu=self.gpu,
        min_memory_gb=self.min_memory,
        depends_on=dep_ids
    )

    return TronFuture(job_id)