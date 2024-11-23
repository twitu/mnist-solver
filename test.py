from train import train_and_test


def test_model_accuracy():
    # Run training and testing
    train_and_test()

    # Get the final accuracy from the last line of output
    # Note: This assumes your train_and_test function returns the final accuracy
    final_accuracy = train_and_test()

    # Assert accuracy is above 95%
    assert (
        final_accuracy >= 95.0
    ), f"Model accuracy {final_accuracy}% is below required 95%"
