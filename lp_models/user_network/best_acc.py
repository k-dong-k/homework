import torch.nn as nn
from transformation import TPS_SpatialTransformerNetwork

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        양방향 LSTM (BiLSTM) 모델을 정의하는 클래스
        
        :param input_size: 입력 벡터의 크기
        :param hidden_size: LSTM의 은닉 상태 크기
        :param output_size: 최종 출력 벡터의 크기
        """
        super(BidirectionalLSTM, self).__init__()
        
        # 양방향 LSTM 레이어 정의
        self.rnn = nn.LSTM(
            input_size,        # 입력 크기
            hidden_size,       # 은닉 상태 크기
            bidirectional=True, # 양방향 LSTM 사용
            batch_first=True    # 입력 텐서의 첫 번째 차원이 배치 크기
        )
        
        # LSTM의 출력을 output_size 크기로 변환하는 선형 레이어
        self.linear = nn.Linear(hidden_size * 2, output_size)  # 양방향이므로 hidden_size * 2

    def forward(self, input):
        """
        순전파(Forward) 연산을 수행하는 함수
        
        :param input: 입력 텐서 (batch_size x T x input_size)
        :return: LSTM을 거친 최종 출력 텐서 (batch_size x T x output_size)
        """
        try:
            # Multi-GPU 환경에서 LSTM 연산 최적화를 위해 필요함
            self.rnn.flatten_parameters()
        except:
            # 양자화(Quantization)된 모델에서는 flatten_parameters()가 지원되지 않으므로 예외 처리
            pass
        
        # LSTM 레이어를 통과한 결과 (batch_size x T x (2*hidden_size))
        recurrent, _ = self.rnn(input)  
        
        # 선형 변환을 거쳐 최종 출력 크기로 변환 (batch_size x T x output_size)
        output = self.linear(recurrent)  
        
        return output
    

class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=256):
        """
        VGG 기반 특징 추출기
        
        :param input_channel: 입력 이미지의 채널 수 (예: RGB는 3)
        :param output_channel: 출력 채널 크기 (기본값 256)
        """
        super(VGG_FeatureExtractor, self).__init__()
        
        # 출력 채널 크기 설정
        self.output_channel = [
            int(output_channel / 8),  # 첫 번째 컨볼루션 레이어의 출력 채널 (32)
            int(output_channel / 4),  # 두 번째 컨볼루션 레이어의 출력 채널 (64)
            int(output_channel / 2),  # 세 번째 컨볼루션 레이어의 출력 채널 (128)
            output_channel           # 네 번째 컨볼루션 레이어의 출력 채널 (256)
        ]
        
        # CNN 기반 특징 추출 네트워크 정의
        self.ConvNet = nn.Sequential(
            # 첫 번째 컨볼루션 블록
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 출력 크기: (32, H/2, W/2)
            
            # 두 번째 컨볼루션 블록
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 출력 크기: (64, H/4, W/4)
            
            # 세 번째 컨볼루션 블록
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # (128, H/4, W/4)
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 출력 크기: (128, H/8, W/4)
            
            # 네 번째 컨볼루션 블록
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 출력 크기: (256, H/8, W/4)
            
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 출력 크기: (256, H/16, W/4)
            
            # 마지막 컨볼루션 블록
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True)  # 출력 크기: (256, H/16, W/4 - 1)
        )

    def forward(self, input):
        """
        순전파(Forward) 연산을 수행하는 함수
        
        :param input: 입력 이미지 텐서 (batch_size x C x H x W)
        :return: 특징 맵 출력 텐서
        """
        return self.ConvNet(input)


import torch.nn as nn
import torch.nn.functional as F


class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """
    
    def __init__(self, input_channel, output_channel=512):
        """
        VGG 기반 특징 추출기
        
        :param input_channel: 입력 이미지의 채널 수 (예: RGB는 3)
        :param output_channel: 출력 채널 크기 (기본값 512)
        """
        super(VGG_FeatureExtractor, self).__init__()
        
        # 출력 채널 크기 설정
        self.output_channel = [
            int(output_channel / 8),  # 첫 번째 컨볼루션 레이어의 출력 채널 (64)
            int(output_channel / 4),  # 두 번째 컨볼루션 레이어의 출력 채널 (128)
            int(output_channel / 2),  # 세 번째 컨볼루션 레이어의 출력 채널 (256)
            output_channel           # 네 번째 컨볼루션 레이어의 출력 채널 (512)
        ]  
        
        # CNN 기반 특징 추출 네트워크 정의
        self.ConvNet = nn.Sequential(
            # 첫 번째 컨볼루션 블록
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 출력 크기: (64, H/2, W/2)
            
            # 두 번째 컨볼루션 블록
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 출력 크기: (128, H/4, W/4)
            
            # 세 번째 컨볼루션 블록
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # (256, H/4, W/4)
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 출력 크기: (256, H/8, W/4)
            
            # 네 번째 컨볼루션 블록
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 출력 크기: (512, H/8, W/4)
            
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 출력 크기: (512, H/16, W/4)
            
            # 마지막 컨볼루션 블록
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True)  # 출력 크기: (512, H/16, W/4 - 1)
        )

    def forward(self, input):
        """
        순전파(Forward) 연산을 수행하는 함수
        
        :param input: 입력 이미지 텐서 (batch_size x C x H x W)
        :return: 특징 맵 출력 텐서
        """
        return self.ConvNet(input)

    
class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """
    
    def __init__(self, input_channel, output_channel=512):
        """
        ResNet 기반 특징 추출기
        
        :param input_channel: 입력 이미지의 채널 수
        :param output_channel: 출력 채널 크기 (기본값 512)
        """
        super(ResNet_FeatureExtractor, self).__init__()
        
        # ResNet 구조의 CNN을 사용하여 특징을 추출
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])
    
    def forward(self, input):
        """
        순전파(Forward) 연산을 수행하는 함수
        
        :param input: 입력 이미지 텐서 (batch_size x C x H x W)
        :return: 특징 맵 출력 텐서
        """
        return self.ConvNet(input)

class GRCL(nn.Module):
    """ Gated Recurrent Convolutional Layer (GRCL) for Gated RCNN """
    
    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        """
        GRCL 초기화
        
        :param input_channel: 입력 채널 수
        :param output_channel: 출력 채널 수
        :param num_iteration: GRCL 반복 횟수
        :param kernel_size: 컨볼루션 커널 크기
        :param pad: 패딩 크기
        """
        super(GRCL, self).__init__()
        
        # 입력 특징 맵을 게이트 함수(G)로 변환하는 1x1 컨볼루션
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        # 현재 상태의 특징 맵을 게이트 함수(G)로 변환하는 1x1 컨볼루션
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        # 입력 특징 맵을 피드포워드 경로로 변환하는 컨볼루션
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        # 이전 상태 특징 맵을 피드포워드 경로로 변환하는 컨볼루션
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)
        
        # 초기 특징 맵에 대한 배치 정규화
        self.BN_x_init = nn.BatchNorm2d(output_channel)
        
        self.num_iteration = num_iteration
        
        # 반복적으로 적용될 GRCL 유닛을 생성
        self.GRCL = nn.Sequential(*[GRCL_unit(output_channel) for _ in range(num_iteration)])
    
    def forward(self, input):
        """
        순전파(Forward) 연산 수행
        
        :param input: 입력 이미지 텐서
        :return: GRCL을 통과한 출력 텐서
        """
        # 입력 데이터로부터 게이트 함수 및 피드포워드 특징 맵을 계산
        wgf_u = self.wgf_u(input)  # 게이트용 변환
        wf_u = self.wf_u(input)  # 피드포워드 변환
        
        # 초기 특징 맵에 배치 정규화 및 활성화 함수 적용
        x = F.relu(self.BN_x_init(wf_u))
        
        # GRCL 유닛을 반복적으로 적용하여 최종 특징 맵 계산
        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))
        
        return x

class GRCL_unit(nn.Module):
    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)  # 입력 특징 맵의 게이트 배치 정규화
        self.BN_grx = nn.BatchNorm2d(output_channel)  # 이전 상태의 게이트 배치 정규화
        self.BN_fu = nn.BatchNorm2d(output_channel)  # 입력 특징 맵의 배치 정규화
        self.BN_rx = nn.BatchNorm2d(output_channel)  # 이전 상태의 배치 정규화
        self.BN_Gx = nn.BatchNorm2d(output_channel)  # 게이트 적용 후의 배치 정규화

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)  # 입력 특징 맵을 기반으로 한 게이트 값 계산
        G_second_term = self.BN_grx(wgr_x)  # 이전 상태를 기반으로 한 게이트 값 계산
        G = torch.sigmoid(G_first_term + G_second_term)  # 게이트 활성화 함수 적용

        x_first_term = self.BN_fu(wf_u)  # 입력 특징 맵의 변환 값 계산
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)  # 이전 상태의 변환 값과 게이트 결합
        x = F.relu(x_first_term + x_second_term)  # ReLU 활성화 함수 적용

        return x

class BasicBlock(nn.Module):
    expansion = 1  # 기본 블록에서는 확장 계수 없음

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)  # 첫 번째 3x3 컨볼루션 레이어
        self.bn1 = nn.BatchNorm2d(planes)  # 첫 번째 배치 정규화
        self.conv2 = self._conv3x3(planes, planes)  # 두 번째 3x3 컨볼루션 레이어
        self.bn2 = nn.BatchNorm2d(planes)  # 두 번째 배치 정규화
        self.relu = nn.ReLU(inplace=True)  # 활성화 함수 적용
        self.downsample = downsample  # 다운샘플링 레이어 (옵션)
        self.stride = stride  # 스트라이드 설정

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """ 3x3 컨볼루션 필터 생성 """
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x  # 스킵 연결을 위한 원본 입력 저장

        out = self.conv1(x)  # 첫 번째 컨볼루션 연산 수행
        out = self.bn1(out)  # 첫 번째 배치 정규화 적용
        out = self.relu(out)  # ReLU 활성화 함수 적용

        out = self.conv2(out)  # 두 번째 컨볼루션 연산 수행
        out = self.bn2(out)  # 두 번째 배치 정규화 적용

        if self.downsample is not None:
            residual = self.downsample(x)  # 다운샘플링이 필요한 경우 적용
        out += residual  # 스킵 연결 수행 (잔차 추가)
        out = self.relu(out)  # ReLU 활성화 함수 적용

        return out



class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        # 출력 채널 크기 설정 (각 블록의 출력 채널을 조정)
        self.output_channel_block = [
            int(output_channel / 4),  # 첫 번째 블록 출력 채널
            int(output_channel / 2),  # 두 번째 블록 출력 채널
            output_channel,          # 세 번째 블록 출력 채널
            output_channel           # 네 번째 블록 출력 채널
        ]

        # 입력 특징맵의 채널 크기 설정
        self.inplanes = int(output_channel / 8)
        
        # 초기 컨볼루션 계층 정의
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 첫 번째 풀링 및 ResNet 블록
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        # 두 번째 풀링 및 ResNet 블록
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        # 세 번째 풀링 및 ResNet 블록
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        # 네 번째 블록 및 최종 컨볼루션 계층
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        ResNet 블록을 생성하는 함수
        
        :param block: 블록 유형 (BasicBlock 또는 BottleneckBlock)
        :param planes: 출력 채널 크기
        :param blocks: 해당 레이어에서 사용될 블록 수
        :param stride: 스트라이드 값 (기본값 1)
        :return: 생성된 ResNet 블록
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        순전파(Forward) 연산을 수행하는 함수
        
        :param x: 입력 이미지 텐서 (batch_size x C x H x W)
        :return: 특징 맵 출력 텐서
        """
        x = self.conv0_1(x)  # 첫 번째 컨볼루션 수행
        x = self.bn0_1(x)  # 배치 정규화 수행
        x = self.relu(x)  # ReLU 활성화 함수 적용
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)  # 첫 번째 최대 풀링 수행
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)  # 두 번째 최대 풀링 수행
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)  # 세 번째 최대 풀링 수행
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x





class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        
        """ 특징 추출 (Feature Extraction) """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        
        """ 공간 변환 네트워크 (Spatial Transformer Network) """
        self.Transformation = TPS_SpatialTransformerNetwork(F=20, I_size=(60, 200), I_r_size=(60,200), I_channel_num=1)
        
        # self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)  # 다른 특징 추출기 사용 가능

        # 특징 추출기의 출력 채널 크기 저장
        self.FeatureExtraction_output = output_channel
        
        # 적응형 평균 풀링 레이어 (세로 크기를 1로 만듦)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ 시퀀스 모델링 (Sequence Modeling) """
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),  # 양방향 LSTM 첫 번째 레이어
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)  # 양방향 LSTM 두 번째 레이어
        )
        
        # 시퀀스 모델링의 출력 크기 설정
        self.SequenceModeling_output = hidden_size

        """ 문자 예측 (Prediction) """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)  # 최종 예측을 위한 선형 계층

    def forward(self, input, text):
        """ 순전파 (Forward) 연산 수행 """
        
        # 공간 변환 네트워크 적용 (입력 이미지 정렬)
        input = self.Transformation(input)
        
        """ 특징 추출 단계 """
        visual_feature = self.FeatureExtraction(input)  # VGG를 이용한 특징 추출
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # 채널 차원 변경 후 평균 풀링 적용
        visual_feature = visual_feature.squeeze(3)  # 3번 차원을 제거하여 (batch, width, channel) 형태로 변환

        """ 시퀀스 모델링 단계 """
        contextual_feature = self.SequenceModeling(visual_feature)  # LSTM을 이용하여 문맥 정보 학습

        """ 문자 예측 단계 """
        prediction = self.Prediction(contextual_feature.contiguous())  # 선형 계층을 이용한 최종 문자 예측

        return prediction

class CRNNWithSTN(nn.Module):
    def __init__(self, num_classes):
        super(CRNNWithSTN, self).__init__()
        # STN을 모델의 첫 번째 부분에 추가
        self.stn = SpatialTransformerNetwork()
        
        # VGG, ResNet, GRCL과 같은 특징 추출기 추가
        self.vgg_extractor = VGG_FeatureExtractor()
        self.resnet_extractor = ResNet_FeatureExtractor()
        self.grcl_extractor = GRCL()
        
        # BiLSTM을 위한 설정
        self.rnn = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True)
        
        # 출력층
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # STN을 통해 입력 이미지 변환
        x = self.stn(x)
        
        # 특징 추출기 통과
        features_vgg = self.vgg_extractor(x)
        features_resnet = self.resnet_extractor(x)
        features_grcl = self.grcl_extractor(x)
        
        # LSTM에 입력할 수 있도록 차원 조정
        x = features_vgg.view(features_vgg.size(0), -1, 256)
        
        # BiLSTM 통과
        rnn_out, _ = self.rnn(x)
        
        # 최종 출력층
        out = self.fc(rnn_out[:, -1, :])  # 마지막 타임스텝의 출력
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        # Spatial transformer network의 주요 요소들
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 8 * 8, 32),  # Feature map의 크기를 32로 변환
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)  # 2D affine transformation matrix
        )
        
        # 기본적인 2D affine transformation matrix 초기화
        self.init_weights()

    def init_weights(self):
        # Localization network의 가중치 초기화
        for m in self.localization:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        
    def forward(self, x):
        # Feature map에서 affine transformation 파라미터를 계산
        x = self.localization(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layer
        theta = self.fc_loc(x)
        theta = theta.view(-1, 2, 3)  # 2x3 Affine Transformation Matrix
        
        # Affine grid를 생성하고 이미지를 변환
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
