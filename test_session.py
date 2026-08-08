from src.core.session_manager import SessionManager

manager = SessionManager()

session = manager.get_session()

logger = manager.get_logger()

logger.start()

session.project_name = "Sample AI Project"

logger.log(

    "Analysis",

    "Project Loaded",

    "SUCCESS"

)

session.language = "Python"

logger.log(

    "Detection",

    "ONNX Model Found",

    "SUCCESS"

)

print()

print(session)