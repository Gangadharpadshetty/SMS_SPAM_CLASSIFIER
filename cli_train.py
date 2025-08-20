from src.train import train_and_evaluate


def main() -> None:
    metrics = train_and_evaluate()
    print(metrics)


if __name__ == "__main__":
    main()


