# EvaluateModel.py (새로 만들 파일)

import numpy
import pandas
import sklearn # 스케일링 및 모델, 교차 검증 함수 사용
# from sklearn.model_selection import cross_val_score # cross_val_score 함수 불러옴
from sklearn.model_selection import StratifiedKFold # StratifiedKFold 사용 추천 (장르별 비율 유지)
from sklearn.preprocessing import MinMaxScaler # 스케일링

# 사용할 모델들
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from config import CreateDataset, Model # 데이터 파일 이름, 모델 설정 등 가져옴

def main():
    # 1. 통합 데이터셋 파일 로드 (CreateDataset.py가 이미 만들었다고 가정)
    try:
        # config에 정의된 데이터셋 이름 사용
        data_set = pandas.read_csv(CreateDataset.NAME, index_col=False)
        print(f"데이터셋 파일 '{CreateDataset.NAME}' 로드 완료. 형태: {data_set.shape}")
    except FileNotFoundError:
        print(f"오류: 데이터셋 파일 '{CreateDataset.NAME}'을 찾을 수 없습니다.")
        print("CreateDataset.py를 실행해서 통합 데이터셋 파일을 먼저 만들어야 합니다.")
        return

    # 2. 데이터셋에서 특징(X)과 레이블(y) 분리
    data_set_numpy = numpy.array(data_set)
    number_of_cols = data_set_numpy.shape[1]
    X = data_set_numpy[:, :number_of_cols - 1] # 특징 데이터
    y = data_set_numpy[:, number_of_cols - 1] # 장르 레이블

    print(f"특징 데이터 X 형태: {X.shape}")
    print(f"레이블 데이터 y 형태: {y.shape}")

    # 3. 특징 스케일링
    # 교차 검증 과정 내에서 폴드별로 스케일링하는 것이 가장 정확하지만,
    # 여기서는 전체 데이터를 미리 스케일링하는 방식으로 간소화.
    # 이상적인 Cross-validation 튜닝에서는 스케일링도 파이프라인에 포함시켜야 함.
    print("전체 데이터 스케일링 시작...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    print("전체 데이터 스케일링 완료.")


    # 4. 사용할 모델 선택 및 객체 생성
    # 논문에서 사용한 모델과 하이퍼파라미터로 설정 (예시: SVM)
    # 이 하이퍼파라미터는 튜닝을 통해 최적값을 찾아야 함!
    #model_to_evaluate = RandomForestClassifier(n_estimators=10)
   # model_to_evaluate = MLPClassifier(hidden_layer_sizes=(100,100), max_iter = 400)
    #model_to_evaluate = KNeighborsClassifier(n_neighbors=Model.NEIGHBOURS_NUMBER)
    model_to_evaluate = SVC(C=100.0, gamma=0.08)
    print(f"\n모델: {type(model_to_evaluate).__name__} ({model_to_evaluate.get_params()})")

    # 5. 10-fold Cross-validation 설정
    # StratifiedKFold는 각 폴드에 클래스(장르) 비율을 비슷하게 유지해줘서 분류 문제에 더 적합해
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10개 폴드, 데이터를 섞어서 분할, 재현성을 위해 random_state 설정

    print("\n10-Fold Cross-validation 시작...")

    # 6. Cross-validation 수행
    # cross_val_score 함수는 cv 객체를 받아서 각 폴드의 성능 점수를 계산해줌
    # scoring: 평가 지표 (accuracy, precision, recall, f1 등 지정 가능)
    scores = sklearn.model_selection.cross_val_score(
        estimator=model_to_evaluate, # 평가할 모델
        X=X_scaled,                  # 특징 데이터 (스케일링된 것 사용)
        y=y,                         # 레이블 데이터
        cv=kf,                       # 교차 검증 폴드 설정
        scoring='accuracy',          # 평가 지표: 정확도
        n_jobs=-1                    # 사용 가능한 모든 CPU 코어 사용해서 빠르게!
    )

    print("10-Fold Cross-validation 완료!")

    # 7. 결과 출력
    print("\n=== 10-Fold Cross-validation 결과 ===")
    print(f"각 폴드의 정확도 점수: {scores}")
    print(f"평균 정확도: {scores.mean():.4f}")
    print(f"정확도의 표준편차: {scores.std():.4f}")
    print("="*40 + "\n")

    # 참고: 각 폴드별 더 상세한 보고서 (Precision, Recall 등)를 보려면 cross_validate 함수를 사용하고
    # return_estimator=True로 설정해서 학습된 모델을 가져온 후 predict해서 classification_report를 만들어야 함.
    # cross_val_score는 간단히 점수만 반환함.


if __name__ == '__main__':
    main()
