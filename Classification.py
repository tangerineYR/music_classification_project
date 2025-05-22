# Classification.py 파일 수정 제안 (GMM 분류기 예측 및 평가)

import joblib
import sklearn
import os
import numpy
import pandas # 사용하지 않지만 import 유지 (config 때문)
import librosa # 오디오 파일 로드 및 경로 찾기

# scikit-learn에서 평가 지표 관련 함수 불러오기
from sklearn.metrics import classification_report

# 필요한 함수와 설정값 가져오기
from Source.Utilities import extract_features # 특징 추출 함수
from Source.Utilities import get_subdirectories # 폴더 목록 가져오는 함수 (Utilities에 있다고 가정)
# get_subdirectories 함수가 만약 CreateDataset.py에만 있다면 Utilities.py로 옮겨서 사용하거나
# 아래 코드에서 os.listdir, os.path.isdir 조합으로 직접 구현해야 해.
from config import Test, Model, CreateDataset # 테스트 경로, 모델 이름, 장르 목록 설정값

from sklearn.mixture import GaussianMixture

def main():
    # 테스트 파일의 실제 장르 (정답)를 저장할 리스트
    true_labels = []
    # 테스트 파일의 특징 데이터를 저장할 리스트
    test_data_list = []

    # config에 정의된 전체 장르 목록 (이 순서대로 보고서 출력될 거야)
    all_genres = CreateDataset.Genres

    # 테스트 데이터 폴더 경로 (예: C:\...\TestFiles)
    test_data_root_dir = Test.DATA_PATH

    # 1. 테스트 데이터 폴더 안의 장르 폴더들을 순회하며 파일 로드 및 특징 추출
    try:
        test_genre_dirs = get_subdirectories(test_data_root_dir)
        # config에 정의된 장르 목록에 없는 테스트 폴더는 건너뛸 수도 있게 필터링 (선택 사항)
        test_genre_dirs = [g for g in test_genre_dirs if g in all_genres]
    except Exception as e:
        print(f"오류: 테스트 데이터 폴더 '{test_data_root_dir}'에서 하위 폴더(장르)를 읽어오지 못했습니다.")
        print(f"오류 내용: {e}")
        return # 오류 발생 시 종료

    if not test_genre_dirs:
         print(f"경고: 테스트 데이터 폴더 '{test_data_root_dir}'에 장르 폴더가 없습니다.")
         print("폴더 구조를 확인해주세요 (예: TestFiles/Blues, TestFiles/Rock 등).")
         return

    # 각 장르 폴더를 돌면서 오디오 파일 로드 및 특징 추출 (GTZAN처럼 파일 1=곡 1)
    for genre in test_genre_dirs:
        genre_path = os.path.join(test_data_root_dir, genre)

        # 해당 장르 폴더 안의 모든 오디오 파일 경로 찾기 (.wav, .au 등)
        audio_files_in_genre = librosa.util.find_files(genre_path)

        if not audio_files_in_genre:
            continue # 파일이 없으면 이 장르는 건너뛰기

        for audio_file_path in audio_files_in_genre:
            try:
                # 오디오 파일 로드 (config의 샘플링 레이트, 5초 길이 사용)
                signal, sr = librosa.load(audio_file_path, sr=CreateDataset.SAMPLING_RATE, duration=5.0)

                # 특징 추출
                features = extract_features(signal)

                # 추출한 특징과 해당 파일의 실제 장르(정답) 저장
                test_data_list.append(features)
                true_labels.append(genre) # 이 파일의 정답은 현재 장르 이름!

            except Exception as e:
                # 파일을 읽거나 처리하다 문제가 생기면 에러 메시지 출력하고 건너뛰기
                print(f"오류 발생: 파일 '{audio_file_path}' 처리 중 오류 - {e}. 이 파일은 건너뜁니다.")
                continue # 다음 파일로 넘어가

    # 모든 테스트 파일 처리가 끝난 후...
    if not test_data_list:
        print("오류: 처리 가능한 테스트 오디오 파일이 없어 예측을 수행할 수 없습니다.")
        return # 데이터 없으면 종료

    # 특징 리스트와 정답 리스트를 numpy 배열로 변환
    test_data_numpy = numpy.array(test_data_list)
    true_labels_numpy = numpy.array(true_labels) # 정답 레이블도 numpy 배열로!

    # 2. 특징 스케일링 (학습 데이터 스케일러 로드하는 게 맞지만, 일단 현재 코드 구조를 따름)
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    test_data_scaled = scaler.fit_transform(test_data_numpy) # 테스트 데이터만으로 스케일링 적용


    # 3. 학습된 GMM 모델들 불러오기 (딕셔너리 형태)
    try:
        # TrainModel.py에서 저장한 GMM 모델 딕셔너리를 불러와
        trained_gmms = joblib.load(Model.NAME) # config에서 모델 파일 이름 가져옴
        print(f"모델 파일 '{Model.NAME}' 로드 완료!")
        # 불러온 모델이 GMM 딕셔너리가 맞는지 간단히 확인
        if not isinstance(trained_gmms, dict) or not all(isinstance(gmm, GaussianMixture) for gmm in trained_gmms.values()):
             print(f"경고: 로드된 모델 파일 '{Model.NAME}'의 형식이 예상(GMM 딕셔너리)과 다를 수 있습니다.")

    except FileNotFoundError:
        print(f"오류: 모델 파일 '{Model.NAME}'을 찾을 수 없습니다.")
        print("TrainModel.py를 먼저 실행해서 GMM 모델들을 학습시키고 저장해야 합니다.")
        return # 모델 파일 없으면 함수 종료
    except Exception as e:
        print(f"오류: 모델 파일 '{Model.NAME}'을 불러오는 중 오류 발생 - {e}")
        return

    # 4. 테스트 데이터 예측 (GMM 로직!)
    predicted_labels = [] # 예측된 장르 이름을 저장할 리스트

    # 테스트 데이터 샘플 하나씩 가져와서 각 GMM 모델에서의 확률 계산
    print("테스트 데이터로 장르 예측 시작 (GMM)...")
    for test_sample_scaled in test_data_scaled:
        # 현재 테스트 샘플에 대한 각 장르 GMM 모델의 로그 확률을 저장할 딕셔너리
        log_likelihoods = {}

        # 학습된 각 장르별 GMM 모델을 순회
        for genre, gmm_model in trained_gmms.items():
            # 주의: gmm.score_samples 함수는 입력으로 2차원 배열을 기대해.
            # 테스트 샘플은 현재 1차원 numpy 배열 형태이므로, [test_sample_scaled] 형태로 감싸서 2차원으로 만들어 줘야 해.
            try:
                 # 현재 테스트 샘플이 이 장르의 GMM에서 나올 로그 확률 계산
                 log_prob = gmm_model.score_samples([test_sample_scaled]) # 결과는 1개짜리 numpy 배열
                 log_likelihoods[genre] = log_prob[0] # 로그 확률 값을 딕셔너리에 저장

            except ValueError as ve:
                 # 모델 fit 시 사용한 특징 차원과 예측할 데이터의 특징 차원이 다를 때 발생하는 오류
                 print(f"오류: 예측할 데이터의 특징 차원이 모델과 다릅니다. ({ve})")
                 print(f"데이터 특징 차원: {test_sample_scaled.shape[0]}, 모델 기대 차원: 확인 필요") # 모델 기대 차원은 GMM 객체에서 직접 확인하기 어려움
                 # 이 경우 특징 추출 또는 데이터 준비 과정에 문제가 있을 수 있습니다.
                 return
            except Exception as e:
                 # 기타 예측 중 발생한 오류
                 print(f"오류: 예측 중 예상치 못한 오류 발생 - {e}")
                 # 이 오류가 자주 발생하면 GMM 모델 학습 또는 데이터 자체에 문제가 있을 수 있습니다.
                 # 일단 오류 발생 시 해당 샘플은 예측 불가로 처리하거나 기본값 지정 가능.
                 # 여기서는 오류 나면 루프 중단 대신 그냥 넘어갈게요. 실제 사용 시 처리 방식 고려 필요.
                 print(f"경고: 샘플 예측 중 오류 발생 - {e}. 이 샘플 예측은 건너뜁니다.")
                 # 오류 발생 시 예측값을 None 등으로 처리하고 나중에 필터링 필요할 수 있음
                 log_likelihoods[genre] = -numpy.inf # 오류난 장르는 최소 확률 부여

        # 계산된 로그 확률들 중에서 가장 큰 값을 가진 장르를 찾아서 예측 결과로 결정
        # log_likelihoods 딕셔너리에서 값이 가장 큰 항목의 key(장르 이름)를 찾음
        if log_likelihoods: # 로그 확률이 하나라도 계산되었으면
             predicted_genre = max(log_likelihoods, key=log_likelihoods.get)
        else:
             # 어떤 이유로 모든 장르에서 로그 확률 계산이 실패했을 경우 (발생 가능성 낮음)
             predicted_genre = "unknown" # 예측 실패 처리

        predicted_labels.append(predicted_genre) # 예측된 장르 이름을 리스트에 추가

    predicted_labels_numpy = numpy.array(predicted_labels) # 예측 결과 리스트를 numpy 배열로 변환

    print("장르 예측 완료!")


    # === 모델 평가 결과 출력 (classification_report 사용) ===
    print("\n" + "="*40)
    print("       음악 장르 분류 모델 성능 보고서 (GMM)")
    print("="*40)
    # sklearn.metrics.classification_report 함수 사용
    # 실제 정답, 모델 예측 결과, 장르 목록(all_genres)을 인자로 넘겨주면 됨
    # zero_division=0: 예측 결과에 없는 장르가 있어도 오류 대신 0으로 표시
    report = classification_report(true_labels_numpy, predicted_labels_numpy, target_names=all_genres, zero_division=0)

    print(report)

    print("="*40)
    print("보고서 설명:")
    print("- precision: 모델이 특정 장르로 예측했을 때, 실제로 그 장르인 비율")
    print("- recall: 실제 특정 장르인 샘플 중, 모델이 그 장르로 올바르게 예측한 비율 (장르별 정확도)")
    print("- f1-score: precision과 recall의 조화 평균")
    print("- support: 해당 장르의 실제 샘플 수")
    print("- accuracy: 전체 샘플 중 올바르게 예측한 비율 (전체 정확도)")
    print("- macro avg: 장르별 precision, recall, f1-score의 단순 평균")
    print("- weighted avg: 장르별 precision, recall, f1-score의 샘플 수 가중 평균")
    print("="*40 + "\n")


if __name__ == '__main__':
    main()
