from src.services.optimization_service import OptimizationService


def main():

    print("=" * 60)
    print("ARM Developer AutoPilot")
    print("=" * 60)

    service = OptimizationService()

    session = service.optimize_project(
        "examples/sample_ai_project"
    )

    print()

    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("Project :", session.project_name)
    print("Language:", session.language)

    print()

    print("Detected Models:")

    for model in session.models:

        print(f" • {model['model_name']}")

    print()

    if session.best_result:

        print("Best Configuration")

        print("--------------------")

        for key, value in session.best_result.items():

            print(f"{key}: {value}")


if __name__ == "__main__":

    main()