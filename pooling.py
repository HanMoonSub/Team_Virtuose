import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    """
    GeM (Generalized Mean) Pooling Layer

    GeM pooling은 일반적인 평균/최대 풀링을 일반화한 방식
    입력 feature map을 각 채널마다 공간적으로 "p-평균" 방식으로 pooling
    p가 1이면 평균 풀링, p가 매우 크면 최대 풀링과 유사

    Args:
        p (float): 초기 p 값. 학습 가능한 파라미터로 설정.
        eps (float): 작은 수로, 0으로 나누는 것을 방지.
    """

    def __init__(self, p=2.0, eps=1e-6):
        super(GeM, self).__init__()
      
        # p를 학습 가능한 파라미터로 선언
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Forward pass

        Args:
            x (Tensor): 입력 feature map, shape (B, C, H, W)

        Returns:
            Tensor: GeM pooled feature, shape (B, C, 1, 1)
        """
        return self.gem(x, p=self.p, eps=self.eps)

    @staticmethod
    def gem(x, p, eps):
        """
        실제 GeM 연산 수행

        Steps:
        1. eps로 clamp하여 작은 값 방지
        2. p 제곱
        3. 채널별 global average pooling
        4. 1/p 제곱하여 역변환
        """
        return F.avg_pool2d(
            x.clamp(min=eps).pow(p),  # 각 요소를 p 제곱
            kernel_size=(x.size(-2), x.size(-1)  # 전체 공간 pooling
        ).pow(1.0 / p)

    def __repr__(self):
        # print 시 모델 정보를 보기 좋게 표시
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"


# ======= 사용 예시 =======
if __name__ == "__main__":
    gem = GeM(p=3)
    print(gem)

    # 예제 입력: 배치 2, 채널 4, 8x8 feature map
    x = torch.rand(2, 4, 8, 8)
    out = gem(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)