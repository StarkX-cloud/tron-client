class TronRuntime:

    def gather(self, *futures):

        results = []

        for future in futures:

            result = future.result()

            results.append(result)

        return results