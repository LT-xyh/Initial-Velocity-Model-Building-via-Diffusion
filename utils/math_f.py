import math


def closest_power_of_two(n):
    """
    找到最接近数字n的2的幂（2^k）

    参数:
        n: 输入的正整数

    返回:
        最接近n的2的幂，如果与两个幂距离相等，则返回较大的那个
    """
    if n <= 0:
        raise ValueError("输入必须是正整数")

    # 计算以2为底的对数
    log2_n = math.log2(n)

    # 找到上下两个可能的指数
    lower_exp = math.floor(log2_n)
    upper_exp = lower_exp + 1

    # 计算对应的2的幂
    lower_power = 2 ** lower_exp
    upper_power = 2 ** upper_exp

    # 比较距离
    if n - lower_power <= upper_power - n:
        return lower_power
    else:
        return upper_power


# 测试示例
if __name__ == "__main__":
    test_numbers = [1, 5, 7, 8, 9, 15, 16, 30, 31, 32, 100, 1000]
    for num in test_numbers:
        print(f"{num} 最接近的2的幂是 {closest_power_of_two(num)}")
