import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)
print(arr.ndim)
print(arr.dtype)
print(arr.shape)
print(arr.size)

arr = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
print(arr)
print(arr.ndim)
print(arr.dtype)
print(arr.shape)
print(arr.size)

arr = np.arange(10).reshape(2, 5)
print(arr)

arr = np.random.random((2, 5))
print(arr)

arr = np.random.randn(2, 5)  # 正太分布
print(arr)

arr = np.zeros((2, 3))
arr = np.ones((2, 3))
arr = np.empty((2, 3))
print(arr)

arr = np.arange(10).reshape(2, 5)
print(arr)
print(np.where(arr > 5, arr, 0))

arr = np.random.randint(1, 10, 9).reshape(3, 3)
print(arr)
print(arr[0])  # 第一行
print(arr[0, 0])
print(arr[:, 0])  # 第一列

print()

arr = np.random.randint(1, 10, 20).reshape(4, 5)
print("arr")
print(arr)
print()
# 1,4行
print(arr[0:4:3])
print(arr[[0, 3]])
print(arr[0, 3])

# 1,4列
print(arr[:, [0, 3]])
print(arr[:, 0:4:3])

# 2,3行  2,3列
print("2,3行  2,3列")
print(arr[[1, 2], [1, 2]])  # error
print(arr[[1, 2], 1:3])

print("print it")
for i in range(0, arr.shape[0]):
    for j in range(0, arr.shape[1]):
        print(arr[i, j], end=" ")
    print()

print(arr.sum())
print((arr > 5).sum())#个数
print(arr.sum(0))#列之和
print(arr.sum(1))#行之和

brr = np.random.randint(1, 10, 20).reshape(4, 5)
print(arr)
print(brr)
#print(arr + brr)
print(arr * brr)
#print(np.dot(arr, brr))#error
print(brr.T)
print(np.dot(arr, brr.T))



