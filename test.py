import torch

# 초기값 및 대상 점 y
x = torch.tensor([1.0, 1.0], requires_grad=True)
y = torch.tensor([2.0, 2.0])

# 제약 조건 함수 g(x)
def g(x):
    return - x[:, 0] ** 2 + x[:, 1] + 3

# 최적화 함수 정의 (페널티 함수)
def loss_fn(x, y, mu=10):
    distance = torch.norm(x - y)
    penalty = mu * g(x) ** 2
    return distance + penalty

# 옵티마이저 설정
optimizer = torch.optim.SGD([x], lr=0.01)

# 최적화 반복
for i in range(500):
    optimizer.zero_grad()
    # loss = loss_fn(x, y)
    loss = torch.norm(x - y) + 10 * g(x.unsqueeze(0)) ** 2
    loss.backward()
    optimizer.step()

    # 제약 조건이 거의 만족되면 조기 종료
    if g(x.unsqueeze(0)).abs() < 1e-4:
        break

# 결과 출력
print("최적화 결과:", x.detach().numpy())
print("제약 조건 g(x):", g(x.unsqueeze(0)).item())

print(f"{1e-1:.7f}")
