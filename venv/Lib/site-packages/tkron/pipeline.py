import time


class PipelineTask:

    def __init__(
        self,
        wrapper,
        name=None,
        depends=None
    ):

        self.wrapper = wrapper

        self.name = (
            name
            or wrapper.fn.__name__
        )

        self.depends = depends or []

        self.future = None

        self.started = False

        self.completed = False

        self.result_data = None

        self.start_time = None

        self.end_time = None


class Pipeline:

    def __init__(self):

        self.tasks = []

    # =========================
    # REGISTER TASK
    # =========================

    def task(
        self,
        wrapper,
        name=None,
        depends=None
    ):

        t = PipelineTask(
            wrapper=wrapper,
            name=name,
            depends=depends
        )

        self.tasks.append(t)

        return t

    # =========================
    # STATUS DISPLAY
    # =========================

    def show_live_status(self):

        print("\n==============================")
        print("TRON LIVE PIPELINE STATUS")
        print("==============================")

        for task in self.tasks:

            if task.completed:

                state = "COMPLETED"

            elif task.started:

                state = "RUNNING"

            else:

                state = "WAITING"

            print(
                f"{task.name:<20} | {state}"
            )

        print("==============================\n")

    # =========================
    # EXECUTE PIPELINE
    # =========================

    def run(self):

        print(
            "\n=== TRON PIPELINE START ===\n"
        )

        total_tasks = len(self.tasks)

        completed_count = 0

        while completed_count < total_tasks:

            # -------------------------
            # START READY TASKS
            # -------------------------

            for task in self.tasks:

                if task.started:
                    continue

                deps_ready = True

                for dep in task.depends:

                    if not dep.completed:

                        deps_ready = False
                        break

                if not deps_ready:
                    continue

                print(
                    f"[PIPELINE] STARTING "
                    f"{task.name}"
                )

                task.future = (
                    task.wrapper.submit()
                )

                task.started = True

                task.start_time = time.time()

            # -------------------------
            # CHECK COMPLETIONS
            # -------------------------

            for task in self.tasks:

                if not task.started:
                    continue

                if task.completed:
                    continue

                if task.future.done():

                    task.completed = True

                    task.end_time = time.time()

                    runtime = round(
                        task.end_time
                        - task.start_time,
                        2
                    )

                    task.result_data = (
                        task.future.result()
                    )

                    completed_count += 1

                    print(
                        f"[PIPELINE] COMPLETED "
                        f"{task.name}"
                    )

                    print(
                        f"[PIPELINE] RUNTIME "
                        f"{runtime}s"
                    )

                    print(
                        f"[PIPELINE] RESULT "
                        f"{task.result_data}"
                    )

                    print()

            # -------------------------
            # LIVE VIEW
            # -------------------------

            self.show_live_status()

            time.sleep(1)

        # =========================
        # FINAL SUMMARY
        # =========================

        print(
            "\n=== TRON PIPELINE COMPLETE ===\n"
        )

        print(
            "FINAL EXECUTION SUMMARY:\n"
        )

        for task in self.tasks:

            runtime = round(
                task.end_time
                - task.start_time,
                2
            )

            print(
                f"{task.name:<20} "
                f"| {runtime}s"
            )

        print("\nALL TASKS EXECUTED.\n")