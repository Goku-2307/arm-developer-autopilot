from arm_database import ARM_DATABASE


class ArmAdvisor:

    def __init__(self, model_type):

        self.model_type = model_type

    def generate_plan(self):

        if self.model_type not in ARM_DATABASE:

            return None

        return ARM_DATABASE[self.model_type]