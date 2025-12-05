#############################################################################
# CBFI: Case-Based Feature interaction_effection Library
# Author: Sejong Oh
#  
# Methods             Sescription
# ---------------     -----------------------------------------
# CB_FeatureImp       Create a feature importance table
# CB_Featureinteraction_effect  Create a feature interaction_effection table
# CB_FItable          Create input data for CB_plot.FIgraph
# CB_plot.contribute  Pie chart of feature contribution
# CB_plot.FIgraph     Feature interaction_effection graph
# CB_plot.imp Bar     chart of feature importance
# CB_plot.interaction_effect    Bar chart of feature interaction_effection
# CB_plot.PA          Prediction analysis chart

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


## shuffle feature values ####################################################
def shuffle_feature(feature_series, seed=100):

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(feature_series))
    return feature_series.iloc[idx].reset_index(drop=True)


## calculate interaction_effection of two features###########################################
def CB_Featureinteraction_effect(model, train, cl, F1, F2 = None,
                       itr= 10, task= "regression",
                       parallel = False):
    # Parameters
    # ----------
    # model : 예측 모델. train 데이터에 대해 model.predict(df)를 호출할 수 있어야 함.
    # train : 데이터셋 X .
    # cl    : 실제 레이블 y. 분류라면 범주형, 회귀라면 수치형.
    # F1    : 첫번째 피처 이름.
    # F2    : 두번째 피처 이름. None이면 F1을 제외한 모든 컬럼을 후보로 함.
    # itr   : int반복 횟수. 실제 반복은 pair_Featureinteraction_effect 내부에서 처리될 것.
    # task  : 'regression' 또는 'classification'
    # parallel : True면 병렬 처리. 기본 False. (pair_Featureinteraction_effect 내부 혹은 이 함수에서 병렬 처리)
    #     
    # Returns
    # -------
    # dict
    #     {'task': task, 'data': DataFrame} 형태.
    #     DataFrame 컬럼: class, feature1, feature2,
    #                     interaction_effect, cont_F1, cont_F2, cont_common
 
    # --- 입력 체크 ----------------------------------------------------
    col_names = train.columns.tolist()
    if F1 not in col_names:
        raise ValueError("F1 is wrong!")
    if F2 is not None:
        if F2 not in col_names:
            raise ValueError("F2 is wrong!")
    # --------------------------------------------------------------

    # 결과 저장용 리스트
    f_class = []
    f_name_1 = []
    f_name_2 = []
    f_interaction_effect = []
    f_contribute_F1 = []
    f_contribute_F2 = []
    f_contribute_common = []

    # 후보 변수 결정
    if F2 is None:
        candidate = [c for c in col_names if c != F1]
    else:
        candidate = [F2]

    # --- regression 처리 -------------------------------------------------
    if task == 'regression':
        for i, cand in enumerate(candidate):
            # pair_Featureinteraction_effect를 호출하여 결과 얻기
            result = pair_Featureinteraction_effect(model=model, train=train,   # 요거 먼저 생성해야함
                                          cl=cl,
                                          F1=F1, F2=cand,
                                          itr=itr, task=task,
                                          parallel=parallel)
            f_class.append(np.nan)  # 회귀에서는 클래스 없음
            f_name_1.append(F1)
            f_name_2.append(cand)
            f_interaction_effect.append(result['interaction_effect'])
            f_contribute_F1.append(result['cont_F1'])
            f_contribute_F2.append(result['cont_F2'])
            f_contribute_common.append(result['cont_common'])
            print(f"End {F1} : {cand}")

    else:
        # --- classification 처리 ----------------------------------------
        class_names = cl.unique()
        cnt = 0

        for cname in class_names:
            for cand in candidate:
                result = pair_Featureinteraction_effect(model=model, train=train,
                                              cl=cl,
                                              F1=F1, F2=cand,
                                              itr=itr, task=task,
                                              class_=cname,  # assuming pair function uses class_
                                              parallel=parallel)
                f_class.append(cname)
                f_name_1.append(F1)
                f_name_2.append(cand)
                f_interaction_effect.append(result['interaction_effect'])
                f_contribute_F1.append(result['cont_F1'])
                f_contribute_F2.append(result['cont_F2'])
                f_contribute_common.append(result['cont_common'])
                cnt += 1
            print(f"End class: {cname} ...")

        # 모든 클래스 합산 항목 추가
        for cand in candidate:
            # 해당 candidate에 대해 이전에 저장된 값 합계 계산
            indices = [j for j, x in enumerate(f_name_2) if x == cand]
            interaction_effect_sum = sum(f_interaction_effect[j] for j in indices)
            F1_sum = sum(f_contribute_F1[j] for j in indices)
            F2_sum = sum(f_contribute_F2[j] for j in indices)
            common_sum = sum(f_contribute_common[j] for j in indices)

            f_class.append("_all_")
            f_name_1.append(F1)
            f_name_2.append(cand)
            f_interaction_effect.append(interaction_effect_sum)
            f_contribute_F1.append(F1_sum)
            f_contribute_F2.append(F2_sum)
            f_contribute_common.append(common_sum)
            cnt += 1

    # 결과 데이터프레임 생성
    final = pd.DataFrame({
        "class": f_class,
        "feature1": f_name_1,
        "feature2": f_name_2,
        "interaction_effect": f_interaction_effect,
        "cont_F1": f_contribute_F1,
        "cont_F2": f_contribute_F2,
        "cont_common": f_contribute_common
    })

    return {"task": task, "data": final}

## make a feature interaction_effection table #############################
from itertools import combinations  # 조합 생성

def CB_FItable(model, train, cl, itr=50, task="regression",
               class_="_all_", parallel=False):

    # ---- Error handling ----
    if task not in ("regression", "classification"):
        print("task should be one of 'regression', 'classification'")
        return None

    if task == "classification":
        cname = list(cl.unique()) + ["_all_"]
        if class_ not in cname:
            print("Error. class name is wrong!")
            return None

    # ---- Basic dataset split ----
    train2 = train.copy()    # X

    # ---- All pairwise feature combinations ----
    features = list(train2.columns)
    cbn = list(combinations(features, 2))

    # ---- First pair interaction_effections (initialize) ----
    F1, F2 = cbn[0]
    tmp = CB_Featureinteraction_effect(model, train, cl, F1=F1, F2=F2,
                             itr=itr, task=task, 
                             parallel=parallel)
    interaction_effect = tmp["data"]

    # ---- Remaining pairs ----
    if len(cbn) >= 2:
        for F1, F2 in cbn[1:]:
            tmp = CB_Featureinteraction_effect(model, train, cl,
                                     F1=F1, F2=F2, itr=itr,
                                     task=task)
            interaction_effect = pd.concat([interaction_effect, tmp["data"]], axis=0, ignore_index=True)

    # ---- Keep columns 1:4 as R code ----
    myint = interaction_effect.iloc[:, :4].copy()
    myint.columns = ["class", "from", "to", "weight"]

    # ---- Feature importance ----
    imp = CB_FeatureImp(model, train, cl, itr=itr,
                        task=task)

    myimp = imp["importance"].iloc[:, :3].copy()
    myimp.columns = ["class", "feature", "importance"]

    # ---- Round values ----
    myint["weight"] = myint["weight"].round(3)
    myimp["importance"] = myimp["importance"].round(3)

    return {"Fint": myint, "Fimp": myimp}

####################################################################################
def one_FeatureImp(model, train, cl, seed=100, task="regression"):
    # Parameters
    # ----------
    # model : 예측 모델. train 데이터에 대해 model.predict(df) 호출 가능.
    # train : 전체 데이터셋 X
    # cl    : 실제 레이블 y. 분류라면 범주형, 회귀라면 수치형.
    # seed  : 셔플 시드. 기본 100.
    # task  : "regression" 또는 "classification"

    # Returns
    # -------
    # dict
    #         'task': task,
    #         'overall': DataFrame,
    #         'importance': DataFrame

    # 타깃 제외한 설명 변수
    ds = train.copy()
    N = len(ds)

    # 전체 예측
    pred_all = model.predict(ds)

    # 결과 저장용 리스트
    f_class = []
    f_feature = []
    f_imp_F1 = []
    f_imp_other = []
    f_contribute_F1 = []
    f_contribute_other = []
    f_contribute_common = []
    f_interaction_effect = []

    # 카운터(분류에서 클래스별로 늘어날 때 사용)
    cnt = 0

    # 회귀 분류 공통: 각 피처에 대해 처리
    for i in range(ds.shape[1]):
        # 피처 i를 셔플한 데이터셋 두 종류 준비
        tmp_F1 = ds.copy()
        tmp_other = ds.copy()

        # 첫번째: tmp_F1에서 i열만 셔플
        tmp_F1.iloc[:, i] = shuffle_feature(tmp_F1.iloc[:, i], seed)

        # 두번째: tmp_other에서 i열 제외 나머 모든 열 셔플
        # R 코드: for all k != i, shuffle
        for k in range(ds.shape[1]):
            if k == i:
                continue
            tmp_other.iloc[:, k] = shuffle_feature(tmp_other.iloc[:, k], seed)

        # 예측
        pred_F1 = model.predict(tmp_other)
        pred_other = model.predict(tmp_F1)

        if task == "regression":
            # R의 절대 오차 기반 방식
            F1 = np.abs(pred_F1 - cl.values)
            OTHER = np.abs(pred_other - cl.values)
            ALL = np.abs(pred_all - cl.values)

            f_class.append(np.nan)  # 회귀에서는 클래스 없음
            f_feature.append(ds.columns[i])

            # idx1: positive or negative contribution as per R 조건
            idx1 = np.where(((OTHER > ALL) & (ALL > F1)) |
                            ((F1 > ALL) & (ALL > OTHER)))[0]
            # contribution from F1
            contrib_F1 = np.sum(OTHER[idx1] - ALL[idx1]) / N if len(idx1) > 0 else 0.0
            f_contribute_F1.append(contrib_F1)

            # R 코드에서는 f_contribute_other 를 0으로 설정
            f_contribute_other.append(0.0)

            # idx3: common
            idx3 = np.where((ALL == OTHER) & (OTHER == F1))[0]
            contrib_common = np.sum(OTHER[idx3] - ALL[idx3]) / N if len(idx3) > 0 else 0.0
            f_contribute_common.append(contrib_common)

            # idx4: interaction_effection
            idx4 = np.where(((F1 > ALL) & (OTHER > ALL)) |
                            ((F1 < ALL) & (OTHER < ALL)))[0]
            interaction_effect = np.sum(OTHER[idx4] - ALL[idx4]) / N if len(idx4) > 0 else 0.0
            f_interaction_effect.append(interaction_effect)

            # importance
            f_imp_F1.append(contrib_F1 + interaction_effect)
            f_imp_other.append(0.0 + interaction_effect)

        else:
            # classification
            class_names = np.unique(cl.values)
            for cname in class_names:
                # ensure entry position corresponds to cnt
                # base condition: pred_all == cl == cname
                base = (pred_all == cl.values) & (cl.values == cname)

                # idx1: both wrong but base true -> interaction_effection
                idx1 = np.where((pred_F1 != cl.values) &
                                (pred_other != cl.values) & base)[0]
                interaction_effect = len(idx1) / N

                # idx2: both correct -> common
                idx2 = np.where((pred_F1 == cl.values) &
                                (pred_other == cl.values) & base)[0]
                common = len(idx2) / N

                # idx3: F1 correct, other wrong
                idx3 = np.where((pred_F1 == cl.values) &
                                (pred_other != cl.values) & base)[0]
                contrib_F1 = len(idx3) / N

                # idx4: F1 wrong, other correct
                idx4 = np.where((pred_F1 != cl.values) &
                                (pred_other == cl.values) & base)[0]
                contrib_other = len(idx4) / N

                # store
                f_class.append(cname)
                f_feature.append(ds.columns[i])
                f_interaction_effect.append(interaction_effect)
                f_contribute_common.append(common)
                f_contribute_F1.append(contrib_F1)
                f_contribute_other.append(contrib_other)

                f_imp_F1.append(contrib_F1 + interaction_effect)
                f_imp_other.append(contrib_other + interaction_effect)

                cnt += 1

            # Note: cnt incremented inside loop for each class

    # 회귀일 경우, lists are aligned by feature only
    # 분류일 경우, 클래스별 수치가 추가되었음

    # classification 추가 작업: 모든 클래스 합산 entry "_all_"
    if task != "regression":
        for fs in ds.columns:
            # sum indices where feature == fs
            indices = [j for j, f in enumerate(f_feature) if f == fs]
            # classes for those entries (exist duplicates)
            # sum up their respective metrics
            interaction_effect_sum = sum(f_interaction_effect[j] for j in indices)
            common_sum = sum(f_contribute_common[j] for j in indices)
            F1_sum = sum(f_contribute_F1[j] for j in indices)
            other_sum = sum(f_contribute_other[j] for j in indices)
            imp_F1_sum = sum(f_imp_F1[j] for j in indices)
            imp_other_sum = sum(f_imp_other[j] for j in indices)

            f_class.append("_all_")
            f_feature.append(fs)
            f_interaction_effect.append(interaction_effect_sum)
            f_contribute_common.append(common_sum)
            f_contribute_F1.append(F1_sum)
            f_contribute_other.append(other_sum)
            f_imp_F1.append(imp_F1_sum)
            f_imp_other.append(imp_other_sum)

            cnt += 1

    # 결과 데이터프레임 생성
    overall = pd.DataFrame({
        "class": f_class,
        "feature": f_feature,
        "cont_F1": f_contribute_F1,
        "cont_other": f_contribute_other,
        "cont_common": f_contribute_common,
        "interaction_effect": f_interaction_effect
    })

    importance = pd.DataFrame({
        "class": f_class,
        "feature": f_feature,
        "imp_F1": f_imp_F1,
        "imp_other": f_imp_other
    })

    return {
        "task": task,
        "overall": overall,
        "importance": importance
    }

#####################################################################################
def one_FeatureInt(model, train, cl, F1, F2, seed=100,
                   task = "regression", class_= None):
    # Parameters
    # ----------
    # model : 예측 모델. train 데이터에 대해 model.predict(df) 호출 가능.
    # train : 데이터셋 X
    # cl    : 실제 레이블 y. 분류라면 범주형, 회귀라면 수치형.
    # F1,F2 : 비교할 두 피처 이름.
    # seed  : 셔플 시드. 기본 100.
    # task  : 'regression' 또는 'classification'. 기본 regression.
    # class_: classification일 경우 클래스 이름. 기본 None.

    # Returns
    # -------
    #     {'interaction_effect': ..., 'cont_F1': ..., 'cont_F2': ..., 'cont_common': ...}

    # 설명 변수, 타깃 분리
    ds = train.copy()
    N = len(ds)

    # 전체 예측
    pred_all = model.predict(ds)

    # --- shuffle except F1, F2 -----------------------------------------
    tmp_F1F2 = ds.copy()
    for col in ds.columns:
        if col == F1 or col == F2:
            continue
        tmp_F1F2[col] = shuffle_feature(tmp_F1F2[col], seed)

    # 초기값
    f_contribute_F1 = np.nan
    f_contribute_F2 = np.nan
    f_contribute_common = np.nan
    f_interaction_effect = np.nan

    # 실제 계산 --------------------------------------------------------
    # tmp_F1F2: 다른 컬럼은 셔플, F1/F2는 원본
    tmp_F1 = tmp_F1F2.copy()
    tmp_F2 = tmp_F1F2.copy()

    # F1 셔플
    tmp_F1[F1] = shuffle_feature(tmp_F1[F1], seed)
    # F2 셔플
    tmp_F2[F2] = shuffle_feature(tmp_F2[F2], seed)

    # 예측
    pred_F1 = model.predict(tmp_F2)     # 맞바꾸어 호출: F1이 셔플된 tmp_F2
    pred_F2 = model.predict(tmp_F1)     # F2가 셔플된 tmp_F1
    pred_F1F2 = model.predict(tmp_F1F2) # 둘 다 섞인 데이터

    if task == "regression":
        # 절대 오차를 이용한 계산
        EF1 = np.abs(pred_F1 - cl.values)
        EF2 = np.abs(pred_F2 - cl.values)
        EALL = np.abs(pred_F1F2 - cl.values)

        # idx4 조건: (EF1 > EALL & EF2 > EALL) | (EF1 < EALL & EF2 < EALL)
        cond_pos = (EF1 > EALL) & (EF2 > EALL)
        cond_neg = (EF1 < EALL) & (EF2 < EALL)
        idx4 = np.where(cond_pos | cond_neg)[0]

        if len(idx4) > 0:
            # interaction_effection: average of (EF2-EALL) + (EF1-EALL) over idx4, normalized by 2N
            sum_term = np.sum(EF2[idx4] - EALL[idx4]) + np.sum(EF1[idx4] - EALL[idx4])
            f_interaction_effect = sum_term / (2 * N)
        else:
            f_interaction_effect = 0.0

        # 회귀에서는 f_contribute_* 등은 NA/None으로 그대로 둠

    else:
        # classification
        if class_ is None:
            raise ValueError("For classification task, class_ must be provided.")

        # base: pred_F1F2 == cl & pred_all == cl & cl == class
        base = (pred_F1F2 == cl.values) & (pred_all == cl.values) & (cl.values == class_)

        # idx1: F1 wrong & F2 wrong
        idx1 = np.where((pred_F1 != cl.values) & (pred_F2 != cl.values) & base)[0]
        f_interaction_effect = len(idx1) / N if len(idx1) > 0 else 0.0

        # idx2: both correct
        idx2 = np.where((pred_F1 == cl.values) & (pred_F2 == cl.values) & base)[0]
        f_contribute_common = len(idx2) / N if len(idx2) > 0 else 0.0

        # idx3: F1 correct, F2 wrong
        idx3 = np.where((pred_F1 == cl.values) & (pred_F2 != cl.values) & base)[0]
        f_contribute_F1 = len(idx3) / N if len(idx3) > 0 else 0.0

        # idx4: F1 wrong, F2 correct
        idx4 = np.where((pred_F1 != cl.values) & (pred_F2 == cl.values) & base)[0]
        f_contribute_F2 = len(idx4) / N if len(idx4) > 0 else 0.0

    result = {
        "interaction_effect": f_interaction_effect,
        "cont_F1": f_contribute_F1,
        "cont_F2": f_contribute_F2,
        "cont_common": f_contribute_common
    }

    return result

#####################################################################################
# worker 정의
def worker_1(seed, model, train, cl, F1, F2, task, class_):
    return one_FeatureInt(
        model=model,
        train=train,
        cl=cl,
        F1=F1,
        F2=F2,
        seed=seed,
        task=task,
        class_=class_
    )



def pair_Featureinteraction_effect(model, train, cl, F1, F2,
                         itr= 10, task= "regression",
                         class_ = None,
                         parallel = False):

    # Parameters
    # ----------
    # model 예측 모델. train 데이터에 대해 model.predict(df) 호출 가능.
    # train : 데이터셋 X.
    # cl    : 실제 레이블 y. 분류라면 범주형, 회귀라면 수치형.
    # F1,F2 :  비교할 두 피처 이름.
    # itr   : 반복 횟수. 기본 10.
    # task  : 'regression' 또는 'classification'.
    # class_ : classification 시 클래스 이름 지정. 기본 None.
    # parallel : True면 병렬 처리. 기본 False.
    #
    # Returns
    # -------
    # dict
    #         'interaction_effect': ...,
    #         'cont_F1': ...,
    #         'cont_F2': ...,
    #         'cont_common': ...


    # --- 첫 번째 반복: seed=100 ---
    result = one_FeatureInt(
        model=model,
        train=train,
        cl=cl,
        F1=F1,
        F2=F2,
        seed=100,
        task=task,
        class_=class_
    )

    # --- 반복적으로 더해감 ---
    if itr >= 2:
        if not parallel:
            # 순차 처리
            for i in range(2, itr + 1):
                tmp = one_FeatureInt(
                    model=model,
                    train=train,
                    cl=cl,
                    F1=F1,
                    F2=F2,
                    seed=100 + i,
                    task=task,
                    class_=class_
                )
                # 누적
                result["interaction_effect"] += tmp["interaction_effect"]
                result["cont_F1"] += tmp["cont_F1"]
                result["cont_F2"] += tmp["cont_F2"]
                result["cont_common"] += tmp["cont_common"]
        else:
            # 병렬 처리
            # 사용할 프로세스 수 설정
            n_cores = max(cpu_count() - 1, 1)

            seeds = [100 + i for i in range(2, itr + 1)]
            args_list = [(seed, model, train, cl, F1, F2, task, class_) for seed in seeds]
            with Pool(processes=n_cores) as pool:
                finals = pool.starmap(worker_1, args_list)

            # 누적
            for tmp in finals:
                result["interaction_effect"] += tmp["interaction_effect"]
                result["cont_F1"] += tmp["cont_F1"]
                result["cont_F2"] += tmp["cont_F2"]
                result["cont_common"] += tmp["cont_common"]

        # 평균
        result["interaction_effect"] /= itr
        result["cont_F1"] /= itr
        result["cont_F2"] /= itr
        result["cont_common"] /= itr

    return result

#####################################################################################
# worker 함수
def worker_2(model, train, cl, seed, task):
    return one_FeatureImp(model, train, cl, seed=seed, task=task)


def CB_FeatureImp(model, train, cl, itr=10, task="regression",
                  parallel=False):

    # Parameters
    # ----------
    # model : 예측 모델. train 데이터에 대해 model.predict(df)를 호출할 수 있어야 함.
    # train : 데이터셋 X.
    # cl    : 실제 레이블 y. 분류라면 범주형, 회귀라면 수치형.
    # task  : "regression" 또는 "classification" 
    # parallel : True면 병렬 처리. 기본 False.

    # Returns
    # -------
    # dict
  
    result = one_FeatureImp(model, train, cl, seed=100, task=task)

    if itr >= 2:
        if not parallel:
            # 순차 처리
            for i in range(2, itr + 1):
                tmp = one_FeatureImp(model, train, cl, seed=100 + i, task=task)
                # R 코드에서 result$overall[,3:6] <- result$overall[,3:6] + tmp$overall[,3:6]
                # 파이썬에서는 컬럼 인덱스가 0부터 시작하므로 3:6는 3,4,5 컬럼.
                # 단, one_FeatureImp의 구조 따라 조정 필요.
                result["overall"].iloc[:, 3:6] += tmp["overall"].iloc[:, 3:6].values
                result["importance"].iloc[:, 2:4] += tmp["importance"].iloc[:, 2:4].values
                print("End iteration", i)

        else:
            # 병렬 처리
            # R 코드: detectCores() 후 클러스터 생성. Python은 multiprocessing 사용.
            n_cores = max(cpu_count() - 1, 1)  # 하나는 메인 프로세스용으로 남겨둠
            seeds = [100 + i for i in range(2, itr + 1)]

            args_list = [(model, train, cl, seed, task) for seed in seeds]

            with Pool(processes=n_cores) as pool:
                final = pool.starmap(worker_2, args_list)

            # 합산
            for tmp in final:
                result["overall"].iloc[:, 3:6] += tmp["overall"].iloc[:, 3:6].values
                result["importance"].iloc[:, 2:4] += tmp["importance"].iloc[:, 2:4].values

        # 평균
        result["overall"].iloc[:, 3:6] /= itr
        result["importance"].iloc[:, 2:4] /= itr

    # one_FeatureImp 결과에 class 컬럼이 있어야 함.
    if "class" in result["overall"].columns:
        result["overall"] = result["overall"].sort_values(by="class").reset_index(drop=True)
    if "class" in result["importance"].columns:
        result["importance"] = result["importance"].sort_values(by="class").reset_index(drop=True)

    return result

#####################################################################################
from scipy.stats import mode

def majority(pred_list, true_label):
    # pred_list 중 true_label과 같은 것이 많은지 판단

    pred_arr = np.array(pred_list)
    # true label과 일치하는 것이 가장 많으면 true_label 반환
    counts = np.sum(pred_arr == true_label)
    if counts >= (len(pred_arr) / 2):
        return true_label
    else:
        # 가장 많이 나온 레이블 반환
        return mode(pred_arr, keepdims=True).mode[0]


def generate_data(model, train, cl, F1, F2, itr, task):
    # -----------------------------------------------------
    # 데이터 준비
    # -----------------------------------------------------
    ds = train.copy()

    ds_F1 = ds.copy()
    ds_F2 = ds.copy()
    ds_F1F2 = ds.copy()

    pred_all = model.predict(ds)

    # 예측 저장 (itr 번 + 마지막 1개 컬럼)
    pred_F1 = np.full((len(ds), itr + 1), np.nan)
    pred_F2 = np.full((len(ds), itr + 1), np.nan)
    pred_F1F2 = np.full((len(ds), itr + 1), np.nan)

    # -----------------------------------------------------
    # 반복적으로 셔플 후 predict
    # -----------------------------------------------------
    for i in range(itr):
        # --- F1 keep, 나머지 shuffle ---
        for fs in ds.columns:
            if fs == F1:
                continue
            ds_F1[fs] = shuffle_feature(ds_F1[fs])

        # --- F2 keep, 나머지 shuffle ---
        for fs in ds.columns:
            if fs == F2:
                continue
            ds_F2[fs] = shuffle_feature(ds_F2[fs])

        # --- F1, F2 keep, 나머지 shuffle ---
        for fs in ds.columns:
            if fs in [F1, F2]:
                continue
            ds_F1F2[fs] = shuffle_feature(ds_F1F2[fs])

        pred_F1[:, i] = model.predict(ds_F1)
        pred_F2[:, i] = model.predict(ds_F2)
        pred_F1F2[:, i] = model.predict(ds_F1F2)

    # -----------------------------------------------------
    # majority or mean 계산
    # -----------------------------------------------------
    for i in range(len(ds)):
        if task == "regression":
            pred_F1[i, -1] = np.mean(pred_F1[i, :itr] - cl[i])
            pred_F2[i, -1] = np.mean(pred_F2[i, :itr] - cl[i])
            pred_F1F2[i, -1] = np.mean(pred_F1F2[i, :itr] - cl[i])

        else:  # classification
            pred_F1[i, -1] = majority(pred_F1[i, :itr], cl[i])
            pred_F2[i, -1] = majority(pred_F2[i, :itr], cl[i])
            pred_F1F2[i, -1] = majority(pred_F1F2[i, :itr], cl[i])

    # -----------------------------------------------------
    # 그룹 결정
    # -----------------------------------------------------
    groups = np.array(["ERR"] * len(ds))

    if task == "regression":
        a = pred_F1[:, -1]
        b = pred_F2[:, -1]
        ab = pred_F1F2[:, -1]

        groups[(a <= ab) & (b > ab)] = "G1"
        groups[(a > ab) & (b <= ab)] = "G2"
        groups[(a == ab) & (b == ab)] = "G3"
        groups[(a < ab) & (b < ab)] = "G4n"
        groups[(a > ab) & (b > ab)] = "G4p"

    else:  # classification
        base = (pred_all == cl) & (pred_F1F2[:, -1] == cl)

        groups[(pred_F1[:, -1] == cl) &
               (pred_F2[:, -1] != cl) & base] = "G1"

        groups[(pred_F1[:, -1] != cl) &
               (pred_F2[:, -1] == cl) & base] = "G2"

        groups[(pred_F1[:, -1] == cl) &
               (pred_F2[:, -1] == cl) & base] = "G3"

        groups[(pred_F1[:, -1] != cl) &
               (pred_F2[:, -1] != cl) & base] = "G4"

    return groups


######################################################################################
# Plot methods
#
##################################################################################################
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from IPython.display import display

from plotnine import (
    ggplot, aes, geom_bar, ggtitle, labs, coord_flip, geom_tile,
    theme_void, theme, element_text, scale_fill_discrete, element_text,
    scale_fill_manual, xlim, ylim
)


def CB_plot_imp(result, class_name="_all_", combine=False):
    # Parameters
    # ----------
    # result : one_FeatureImp 또는 CB_FeatureImp 등에서 반환된 구조체.
    #     result['task'] : 'regression' or other (classification)
    #     result['overall'] : pandas.DataFrame
    #     result['importance'] : pandas.DataFrame
    # class_  : classification 일 때 플롯할 클래스 이름. 기본 '_all_'.
    # combine : regression 일 때 importance만 사용하여 단일값으로 그림. 기본 False.
    # figsize : matplotlib figure size.

    task = result["task"]
    overall = result["overall"].copy()
    importance = result["importance"].copy()

    # ----------------------------------------------------
    # Regression
    # ----------------------------------------------------
    if task == "regression":
        overall = overall.iloc[:, 1:]          # remove class column
        importance = importance.iloc[:, 1:]    # remove class column
        stitle = None

        if combine:
            df = importance.iloc[:, [0, 1]].copy()
            df.columns = ["feature", "value"]

            p = (
                ggplot(df, aes(x="reorder(feature, value)", y="value"))
                + geom_bar(stat="identity")
                + ggtitle("Feature Importance", subtitle=stitle)
                + labs(x="feature")
                + coord_flip()
            )
            display(p)

        # regression but combine == False → main_effect + interaction_effect
        cnt = overall.shape[0]
        df1 = pd.DataFrame({
            "feature": overall.iloc[:, 0],
            "value": overall.iloc[:, 1],
            "component": ["main_effect"] * cnt
        })
        df2 = pd.DataFrame({
            "feature": overall.iloc[:, 0],
            "value": overall.iloc[:, 4],
            "component": ["interaction_effect"] * cnt
        })
        df = pd.concat([df1, df2], axis=0)

        p = (
            ggplot(df, aes(x="reorder(feature, value)", y="value", fill="component"))
            + geom_bar(stat="identity")
            + ggtitle("Feature Importance", subtitle=stitle)
            + labs(x="feature")
            + coord_flip()
        )
        display(p)

    # ----------------------------------------------------
    # Classification
    # ----------------------------------------------------
    else:
        cname = list(overall["class"].unique()) + ["_all_"]
        
        if class_name not in cname:
            print("Error. class name is not correct.")
            return None

        # class 선택 후 class column 제거
        overall = overall[overall["class"] == class_name].iloc[:, 1:]
        stitle = f"( class: {class_name} )"

        cnt = overall.shape[0]

        df1 = pd.DataFrame({
            "feature": overall.iloc[:, 0],
            "value": overall.iloc[:, 1],
            "component": ["main_effect"] * cnt
        })
        df2 = pd.DataFrame({
            "feature": overall.iloc[:, 0],
            "value": overall.iloc[:, 4],
            "component": ["interaction_effect"] * cnt
        })
        df = pd.concat([df1, df2], axis=0)

        p = (
            ggplot(df, aes(x="reorder(feature, value)", y="value", fill="component"))
            + geom_bar(stat="identity")
            + ggtitle("Feature Importance", subtitle=stitle)
            + labs(x="feature")
            + coord_flip()
        )

        display(p)

#########################################################################################
def CB_plot_interaction_effect(result, class_="_all_"):
    # Parameters
    # ----------
    # result : dict
    #            'task': 'regression' or 'classification',
    #            'data': pandas.DataFrame  # 컬럼: feature1, feature2, interaction_effect, (class)
    # class_ : 분류일 때 시각화할 클래스 이름. 기본 '_all_'.
    # figsize : tuple, optional
    #     그래프 크기.
 
    df0 = result['data'].copy()

    # ----------------------------
    # Get min/max
    # ----------------------------
    mymin = int(df0['interaction_effect'].min() // 1)   # floor
    mymax = df0['interaction_effect'].max()

    if abs(0 - mymin) < 0.01 or len(df0) == 1:
        mymin = 0

    task = result['task']

    # ----------------------------
    # Classification
    # ----------------------------
    if task == "classification":

        if class_ == "_all_":
            df = df0[df0['class'] != class_].copy()
        else:
            df = df0[df0['class'] == class_].copy()

        df['features'] = df['feature2'] + "-" + df['feature1']

        p = (
            ggplot(df, aes(x='reorder(features, interaction_effect)', y='interaction_effect', fill='class'))
            + geom_bar(stat="identity")
            + ggtitle("Feature interaction_effection",
                      subtitle=f"( class: {class_} )")
            + labs(x="features", y="value")
            + ylim(mymin, mymax)
            + coord_flip()
        )

    # ----------------------------
    # Regression
    # ----------------------------
    else:
        df = df0.copy()
        df['features'] = df['feature2'] + "-" + df['feature1']

        p = (
            ggplot(df, aes(x='reorder(features, interaction_effect)', y='interaction_effect'))
            + geom_bar(stat="identity")
            + ggtitle("Feature interaction_effection")
            + labs(x="features", y="value")
            + ylim(mymin, mymax)
            + coord_flip()
        )

    display(p)

##############################################################################

def CB_plot_contribute(result, class_="_all_"):
    # result: 
    #     "task": "classification",
    #     "data": pandas.DataFrame with columns:
    #         class | F1 | F2 | interaction_effect | cont_F1 | cont_F2 | cont_common

    # --- Validation ---
    if result["task"] != "classification":
        print("This plot is only for classification model")
        return None

    df0 = result["data"]

    if class_ not in df0["class"].unique():
        print("Error. class name is wrong!")
        return None

    # --- Select row for the given class ---
    row = df0[df0["class"] == class_].iloc[0]

    F1 = row["F1"]
    F2 = row["F2"]

    # Construct plotting dataframe
    df = pd.DataFrame({
        "labels": ["interaction_effect", "F1", "F2", "Common"],
        "values": [
            row["interaction_effect"],
            row["cont_F1"],
            row["cont_F2"],
            row["cont_common"]
        ]
    })

    # Matching R factor order: F1, F2, interaction_effect, Common
    df["labels"] = pd.Categorical(
        df["labels"],
        categories=["F1", "F2", "interaction_effect", "Common"],
        ordered=True
    )
    df = df.sort_values("labels")

    subtitle = f"{F1} - {F2} ( class: {class_} )"

    # --- seaborn does not support polar directly, use matplotlib ---
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")

    # Angles for each bar section
    values = df["values"].values
    angles = [0] + list(values.cumsum())
    total = values.sum()
    theta = values / total * 2 * 3.141592653589793

    start = 0
    cmap = sns.color_palette("Set2", len(values))

    for i, (val, label) in enumerate(zip(values, df["labels"])):
        ax.bar(
            x=start,
            height=1,
            width=theta[i],
            bottom=0,
            color=cmap[i],
            edgecolor="white",
            linewidth=1,
            label=label
        )
        start += theta[i]

    # Title + subtitle
    plt.title("Feature Contribution\n" + subtitle, va="bottom")

    # Remove grid & ticks
    ax.set_axis_off()

    # Legend
    plt.legend(title="Components", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


########################################################################
#        
import networkx as nx

def CB_plot_FIgraph(
    FIobj=None,
    task="regression",
    class_name="_all_",
    show_edge_weight=True,
    seed=100,
    th=None
):

    # -----------------------------------------
    # 1. Validation
    # -----------------------------------------
    fint = FIobj["Fint"]
    fimp = FIobj["Fimp"]

    candidate_classes = list(fint["class"].unique()) + ["_all_"]

    if task == "classification":
        if class_name not in candidate_classes:
            print(f"Error! 'class.name' must be one of {candidate_classes}")
            return None

    # -----------------------------------------
    # 2. Select interaction_effection edges & importance nodes
    # -----------------------------------------
    if th is None:
        if task == "classification":
            th = np.mean(np.abs(fint.loc[fint["class"] == class_name, "weight"]))
        else:
            th = np.mean(np.abs(fint["weight"]))

    print("th:", th)

    # filter threshold
    fint2 = fint[np.abs(fint["weight"]) >= th].copy()

    # regression
    if task == "regression":
        edges = fint2.iloc[:, 1:]     # from, to, weight
        nodes = fimp.iloc[:, 1:]      # feature, importance

    # classification
    else:
        edges = fint2[fint2["class"] == class_name].iloc[:, 1:]
        nodes = fimp[fimp["class"] == class_name].iloc[:, 1:]
        nodes = nodes.reset_index(drop=True)

    # -----------------------------------------
    # 3. Build networkx graph
    # -----------------------------------------
    G = nx.Graph()

    # add nodes
    for _, row in nodes.iterrows():
        G.add_node(row["feature"], importance=row["importance"])

    # add edges
    for _, row in edges.iterrows():
        G.add_edge(row["from"], row["to"], weight=row["weight"])

    # -----------------------------------------
    # 4. Node size & color
    # -----------------------------------------
    importance_vals = np.array([G.nodes[n]["importance"] for n in G.nodes()])
    imp_min, imp_max = importance_vals.min(), importance_vals.max()

    norm_imp = (importance_vals - imp_min) / (imp_max - imp_min + 1e-10)
    node_sizes = ((norm_imp / 1.2) + 0.2) * 800  # scale like R (multiply 50)

    # heatmap colors
    cmap = cm.get_cmap("autumn")
    node_colors = [cmap(1 - x) for x in norm_imp]

    # -----------------------------------------
    # 5. Edge width & color
    # -----------------------------------------
    weights = np.array([abs(G[u][v]["weight"]) for u, v in G.edges()])
    wmin, wmax = weights.min(), weights.max()
    ewidth_norm = (weights - wmin) / (wmax - wmin + 1e-10)
    edge_widths = (ewidth_norm + 0.2) * 5  # scale approx

    edge_colors = []
    for u, v in G.edges():
        if G[u][v]["weight"] < 0:
            edge_colors.append("orange")
        else:
            edge_colors.append("gray")

    # -----------------------------------------
    # 6. Draw graph with matplotlib
    # -----------------------------------------
    np.random.seed(seed)
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 10))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black"
    )
    nx.draw_networkx_labels(G, pos, font_size=10)

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors
    )

    # edge weight labels
    if show_edge_weight:
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    title = ""
    if task == "classification":
        title = f"class: {class_name}"
    plt.title(title, fontsize=14)

    plt.axis("off")
    plt.show()

    return G

####################################################################################
def CB_plot_imp(result, class_name="_all_", combine=False):
    # Python version of CB_plot.imp using plotnine.
    
    # result: dict with keys:
    #         - task ('regression' or 'classification')
    #         - overall (DataFrame)
    #         - importance (DataFrame)

    task = result["task"]
    overall = result["overall"].copy()
    importance = result["importance"].copy()

    # ----------------------------------------------------------
    #   Regression
    # ----------------------------------------------------------
    if task == "regression":
        overall = overall.iloc[:, 1:]     # remove 'class' column
        importance = importance.iloc[:, 1:]

        stitle = None

        if combine:
            df = importance.iloc[:, [0, 1]].copy()
            df.columns = ["feature", "value"]

            p = (
                ggplot(df, aes(x="reorder(feature, value)", y="value"))
                + geom_bar(stat="identity")
                + ggtitle("Feature Importance", subtitle=stitle)
                + labs(x="feature")
                + coord_flip()
            )
            return p

        else:
            # regression but not combine → two components (main_effect + interaction_effect)
            cnt = overall.shape[0]

            df1 = pd.DataFrame({
                "feature": overall.iloc[:, 0],
                "value": overall.iloc[:, 1],
                "component": ["main_effect"] * cnt
            })

            df2 = pd.DataFrame({
                "feature": overall.iloc[:, 0],
                "value": overall.iloc[:, 4],
                "component": ["interaction_effect"] * cnt
            })

            df = pd.concat([df1, df2], axis=0)

            p = (
                ggplot(df, aes(x="reorder(feature, value)", y="value", fill="component"))
                + geom_bar(stat="identity")
                + ggtitle("Feature Importance", subtitle=stitle)
                + labs(x="feature")
                + coord_flip()
            )
            return p

    # ----------------------------------------------------------
    #   Classification
    # ----------------------------------------------------------
    else:
        cname = list(result["overall"]["class"].unique()) + ["_all_"]
        if class_name not in cname:
            print("Error. class name is not correct.")
            return None

        # filter selected class and drop class column
        overall = overall[overall["class"] == class_name].iloc[:, 1:]
        stitle = f"( class: {class_name} )"

        cnt = overall.shape[0]

        df1 = pd.DataFrame({
            "feature": overall.iloc[:, 0],
            "value": overall.iloc[:, 1],
            "component": ["main_effect"] * cnt
        })

        df2 = pd.DataFrame({
            "feature": overall.iloc[:, 0],
            "value": overall.iloc[:, 4],
            "component": ["interaction_effect"] * cnt
        })

        df = pd.concat([df1, df2], axis=0)

        p = (
            ggplot(df, aes(x="reorder(feature, value)", y="value", fill="component"))
            + geom_bar(stat="identity")
            + ggtitle("Feature Importance", subtitle=stitle)
            + labs(x="feature")
            + coord_flip()
        )

        display(p)

################################################################################################

def CB_plot_PA(model, train, cl, F1, F2, itr=10,
               task="regression", class_name="_all_"):
    # ------------------------------------------------------
    # Error check
    # ------------------------------------------------------
    feature_names = list(train.columns)

    if F1 not in feature_names :
        print("Name of F1 is wrong..")
        return None

    if F2 not in feature_names:
        print("Name of F2 is wrong..")
        return None

    if task not in ["regression", "classification"]:
        print("task should be one of 'regression' or 'classification'..")
        return None

    if class_name != "_all_":
        if class_name not in cl.unique():
            print("Name of class is wrong..")
            return None

    # ------------------------------------------------------
    print("generate data for plot ....")

    # 사용자 정의 함수 generate.data 가 존재한다고 가정
    groups = generate_data(model, train, cl, F1, F2, itr, task)

    # groups 처리
    groups = pd.Series(groups).astype("object")

    if task == "regression":
        groups.replace("ERR", np.nan, inplace=True)
        levels = ["G1", "G2", "G3", "G4p", "G4n"]
    else:
        levels = ["G1", "G2", "G3", "G4", "ERR"]

    groups = pd.Categorical(groups, categories=levels, ordered=True)

    # 색상
    mycolor = ["#f8766d", "#7cae00", "#c77cff", "#00bfc4", "#fad02c"]

    # 범례 라벨
    if task == "regression":
        legend_label = [
            f"{F1} decrease",
            f"{F2} decrease",
            "No change",
            "Positive interaction_effect",
            "Negative interaction_effect"
        ]
    else:
        legend_label = [
            f"{F1} contribute",
            f"{F2} contribute",
            "Common contribute",
            "interaction_effect",
            "wrong predict"
        ]

    # 데이터 구성
    df = pd.DataFrame({
        F1: train[F1],
        F2: train[F2],
        "groups": groups
    })

    df = df.dropna(subset=["groups"])

    # classification + 특정 class 선택 시 필터링
    if task == "classification" and class_name != "_all_":
        idx = cl == class_name
        df = df[idx]

    # ------------------------------------------------------
    # 셀 width/height 계산
    # ------------------------------------------------------
    x_cell = np.diff(np.sort(df[F1].unique()))
    y_cell = np.diff(np.sort(df[F2].unique()))

    w = np.mean(x_cell) * 5 if len(x_cell) > 0 else 1
    h = np.mean(y_cell) * 3 if len(y_cell) > 0 else 1

    # 해당 그룹 없는 경우 legend 제거
    tbl = df["groups"].value_counts().reindex(levels, fill_value=0)

    zero_idx = np.where(tbl.values == 0)[0]
    if len(zero_idx) > 0:
        legend_label = [lbl for i, lbl in enumerate(legend_label) if i not in zero_idx]
        mycolor = [col for i, col in enumerate(mycolor) if i not in zero_idx]

    # ------------------------------------------------------
    # 제목
    # ------------------------------------------------------
    if class_name != "_all_":
        subtitle = f"{F1} - {F2} (class: {class_name})"
    else:
        subtitle = f"{F1} - {F2}"

    # ------------------------------------------------------
    # plotnine 시각화
    # ------------------------------------------------------
    p = (
        ggplot(df, aes(x=F1, y=F2, fill="groups"))
        + geom_tile(width=w, height=h, na_rm=True)
        + ggtitle("Prediction Analysis", subtitle)
        + scale_fill_manual(values=mycolor, labels=legend_label)
        + labs(x=F1, y=F2)
        + xlim(train[F1].min(), train[F1].max())
        + ylim(train[F2].min(), train[F2].max())
    )

    display(p)

