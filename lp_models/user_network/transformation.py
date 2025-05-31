import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TPS_SpatialTransformerNetwork(nn.Module):
    """ TPS 기반 STN (Spatial Transformer Network)
    입력 이미지를 왜곡 보정하여 OCR 모델의 인식률을 높이는 역할 수행

    Args:
        F (int): 제어점(Fiducial Points) 개수
        I_size (tuple): 원본 이미지 크기 (height, width)
        I_r_size (tuple): 보정된 이미지 크기 (height, width)
        I_channel_num (int): 입력 이미지 채널 수 (기본값: 1)
    """
    
    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super().__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        
        # LocalizationNetwork: 입력 이미지에서 제어점(C') 좌표를 예측하는 네트워크
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        
        # GridGenerator: LocalizationNetwork에서 예측한 C'을 기반으로 보정된 좌표 P'를 생성하는 네트워크
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        """
        batch_I: 입력 이미지 배치 [batch_size x I_channel_num x I_height x I_width]
        반환: 보정된 이미지 배치 [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        
        # Localization Network를 통해 제어점 C'을 예측 (batch_size x F x 2)
        batch_C_prime = self.LocalizationNetwork(batch_I)
        
        # Grid Generator를 통해 변환된 좌표 P'을 생성 (batch_size x n x 2), 여기서 n = I_r_width x I_r_height
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        
        # P'을 [batch_size, I_r_height, I_r_width, 2]로 reshape
        build_P_prime_reshape = build_P_prime.view(batch_C_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2)
        
        # grid_sample을 이용하여 입력 이미지를 P' 좌표를 따라 변환
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """ Localization Network: 입력 이미지에서 제어점(C') 좌표를 예측하는 CNN 네트워크 """

    def __init__(self, F, I_channel_num):
        super().__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        
        # CNN 기반의 특징 추출 네트워크
        self.conv = nn.Sequential(
            nn.Conv2d(I_channel_num, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 출력 크기: (batch_size x 64 x I_height/2 x I_width/2)
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 출력 크기: (batch_size x 128 x I_height/4 x I_width/4)
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 출력 크기: (batch_size x 256 x I_height/8 x I_width/8)
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # 출력 크기: (batch_size x 512 x 1 x 1)
        )
        
        # Fully Connected Layer를 이용한 제어점 좌표 예측
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)  # (batch_size x F*2)
        
        # Fully Connected Layer의 가중치 초기화
        self.localization_fc2.weight.data.fill_(0)
        
        # 초기 제어점 위치를 설정 (RARE 논문의 Fig. 6 참고)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        
        # 초기 제어점 좌표를 bias로 설정
        self.localization_fc2.bias.data = torch.tensor(initial_bias, dtype=torch.float).view(-1)

    def forward(self, batch_I):
        """
        입력: batch_I (배치 이미지) [batch_size x I_channel_num x I_height x I_width]
        출력: batch_C_prime (제어점 좌표) [batch_size x F x 2]
        """
        
        # 배치 크기 가져오기
        batch_size = batch_I.size(0)
        
        # CNN을 통해 특징 추출
        features = self.conv(batch_I).view(batch_size, -1)  # (batch_size x 512)
        
        # Fully Connected Layer를 통과하여 제어점 좌표 예측
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        
        return batch_C_prime


class GridGenerator(nn.Module):
    """ RARE의 Grid Generator로, P에 변환 행렬 T를 곱하여 P_prime을 생성한다. """

    def __init__(self, F, I_r_size):
        """ P_hat 및 inv_delta_C를 사전에 계산하여 저장한다. """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6  # 수치적 불안정을 방지하기 위한 작은 값
        self.I_r_height, self.I_r_width = I_r_size  # 목표 이미지 크기 (height, width)
        self.F = F  # 기준점(fiducial points) 개수
        
        # 기준점 좌표 C 및 보정 대상 그리드 P 생성
        self.C = self._build_C(self.F)  # (F, 2) 형태
        self.P = self._build_P(self.I_r_width, self.I_r_height)  # (n, 2), n = I_r_width * I_r_height
        
        # 변환을 위한 선행 계산값 저장
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # (F+3, F+3)
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # (n, F+3)

    def _build_C(self, F):
        """ 보정된 이미지에서 기준점의 좌표를 생성한다. """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        return np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)  # (F, 2)

    def _build_inv_delta_C(self, F, C):
        """ 변환 행렬을 계산하기 위한 delta_C의 역행렬을 생성한다. """
        hat_C = np.zeros((F, F), dtype=float)
        for i in range(F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        
        delta_C = np.concatenate([
            np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),
            np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
            np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)
        ], axis=0)  # (F+3, F+3)
        
        return np.linalg.inv(delta_C)  # 역행렬 반환 (F+3, F+3)

    def _build_P(self, I_r_width, I_r_height):
        """ 보정된 이미지에서 사용할 그리드 P를 생성한다. """
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        return np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2).reshape([-1, 2])  # (n, 2)

    def _build_P_hat(self, F, C, P):
        """ 변환을 위해 사용될 P_hat을 계산한다. """
        n = P.shape[0]  # 보정된 이미지의 총 픽셀 수 (width * height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # (n, F, 2)
        C_tile = np.expand_dims(C, axis=0)  # (1, F, 2)
        P_diff = P_tile - C_tile  # 거리 계산을 위한 차이 행렬 (n, F, 2)
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2)  # 유클리드 거리 (n, F)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # RBF 커널 적용 (n, F)
        return np.concatenate([np.ones((n, 1)), P, rbf], axis=1)  # (n, F+3)

    def build_P_prime(self, batch_C_prime):
        """ 예측된 기준점을 사용하여 변환된 그리드 P_prime을 생성한다. """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)  # (batch_size, F+3, F+3)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)  # (batch_size, n, F+3)
        
        # 변환 행렬 T 생성을 위한 추가 행렬 (affine 변환 제약 조건 포함)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)), dim=1
        )  # (batch_size, F+3, 2)
        
        # 변환 행렬 T 계산
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # (batch_size, F+3, 2)
        
        # 최종 변환된 그리드 좌표 P_prime 계산
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # (batch_size, n, 2)
        
        return batch_P_prime  # 최종 출력: (batch_size, n, 2)
