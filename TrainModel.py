import numpy
import pandas
import joblib

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

from config import Model, CreateDataset


def main():

    # Read Dataset (CSV file)
    data_set = pandas.read_csv(CreateDataset.NAME, index_col=False)

    # Convert to Array
    data_set = numpy.array(data_set)

    # Calculate Number of Rows and Columns of Dataset File
    number_of_rows, number_of_cols = data_set.shape

    # Get Axis_X and Axis_Y of Data
    data_x = data_set[:, :number_of_cols - 1]
    data_y = data_set[:, number_of_cols - 1]

    # Different Ways of Classification (In Our Project, We Use SVM)
    #model = SVC(C=100.0, gamma=0.08)
    #model = RandomForestClassifier(n_estimators=10)
    #model = MLPClassifier(hidden_layer_sizes=(100,100), max_iter = 400)
    #model = KNeighborsClassifier(n_neighbors=Model.NEIGHBOURS_NUMBER)

    #model.fit(data_x, data_y)

    #joblib.dump(model, Model.NAME)

    
    # === 여기서부터 GMM 학습 로직 시작! ===

    # 학습된 각 장르별 GMM 모델을 저장할 딕셔너리
    # { '장르 이름': 학습된 GMM 모델 객체 } 형태가 될 거야
    trained_gmms = {}

    # config에 정의된 전체 장르 목록을 가져와 (이게 우리가 학습시킬 GMM의 종류)
    genres = CreateDataset.Genres
    print(f"\n총 {len(genres)}개의 장르별 GMM 모델 학습 시작...")

    # 각 장르(클래스)별로 GMM 모델을 학습시키자!
    for genre in genres:
        print(f"'{genre}' 장르 GMM 모델 학습 중...")

        # 1. 현재 장르에 해당하는 데이터만 골라내기
        # data_y 배열에서 현재 genre 이름과 일치하는 샘플들의 인덱스를 찾아서
        # 그 인덱스에 해당하는 data_x의 행들만 선택하는 거야
        genre_data_x = data_x[data_y == genre]

        if genre_data_x.shape[0] == 0:
            print(f"경고: '{genre}' 장르의 데이터가 없습니다. 이 장르의 GMM 모델은 학습하지 않습니다.")
            continue # 해당 장르 데이터가 없으면 건너뛰기

        print(f"  '{genre}' 장르 데이터 형태: {genre_data_x.shape}")

        # 2. GMM 모델 객체 생성
        # n_components: 가우시안 분포 개수 (이게 중요한 하이퍼파라미터!)
        # covariance_type: 공분산 형태 (이것도 중요한 하이퍼파라미터!)
        # random_state: 결과를 재현 가능하게 설정 (실험 시 중요!)
        gmm = GaussianMixture(n_components=5,       # 예시: 가우시안 5개로 모델링
                              covariance_type='full', # 예시: full 공분산 사용
                              random_state=42,
                              max_iter=200)         # 필요시 max_iter 늘려주기 (수렴 경고 시)


        # 3. 현재 장르 데이터로 GMM 모델 학습
        gmm.fit(genre_data_x)

        # 4. 학습된 GMM 모델을 딕셔너리에 저장
        trained_gmms[genre] = gmm

        print(f"  '{genre}' 장르 GMM 모델 학습 완료.")

    print("\n모든 장르 GMM 모델 학습 완료!")
    print(f"학습된 GMM 모델 개수: {len(trained_gmms)}")

    # === 학습된 GMM 모델들을 파일로 저장 ===
    # Model.NAME에 저장하면 기존 분류기 모델 파일(model.pkl)을 덮어쓰게 돼
    # 만약 SVM 모델도 저장하고 GMM 모델도 저장해서 구분하고 싶다면 파일 이름을 다르게 해야 해
    # 예시: joblib.dump(trained_gmms, 'gmm_models.pkl')
    print(f"\n학습된 GMM 모델들을 파일로 저장: {Model.NAME}")
    try:
        joblib.dump(trained_gmms, Model.NAME)
        print("GMM 모델 저장 완료!")
    except Exception as e:
         print(f"오류: GMM 모델 저장 중 오류 발생 - {e}")


if __name__ == '__main__':
    main()
