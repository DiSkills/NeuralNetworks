def relu(v: int) -> int:
    return max(0, v)


def xor(x1: int, x2: int) -> int:
    h1 = relu(x1 + x2 - 1)
    h2 = relu(1 - x1 - x2)
    return (h1 + h2 + 1) % 2


def main():
    for x1 in range(2):
        for x2 in range(2):
            print(x1, x2, xor(x1, x2))


if __name__ == "__main__":
    main()
