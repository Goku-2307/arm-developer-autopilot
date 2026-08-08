from datetime import datetime
import time


class SessionLogger:

    def __init__(self, session):

        self.session = session

        self.stage_start = None

    def start(self):

        self.stage_start = time.perf_counter()

    def log(self,
            stage,
            message,
            status="INFO"):

        duration = 0

        if self.stage_start is not None:

            duration = round(

                time.perf_counter() -

                self.stage_start,

                2

            )

        event = {

            "time": datetime.now().strftime("%H:%M:%S"),

            "stage": stage,

            "status": status,

            "message": message,

            "duration": duration

        }

        self.session.events.append(event)

        print(

            f"[{event['time']}] "

            f"{status:<8}"

            f"{stage:<15}"

            f"{message}"

            f" ({duration}s)"

        )

        self.stage_start = time.perf_counter()