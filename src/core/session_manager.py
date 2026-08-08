from src.core.optimization_session import OptimizationSession
from src.core.session_logger import SessionLogger


class SessionManager:

    def __init__(self):

        self.session = OptimizationSession()

        self.logger = SessionLogger(self.session)

    def get_session(self):

        return self.session

    def get_logger(self):

        return self.logger