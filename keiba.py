def fetch_table(url):

    """出馬表ページから馬名などをDataFrameで取得（簡易版）"""

    try:

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}

        res = requests.get(url, headers=headers, verify=False, timeout=10)

        res.encoding = res.apparent_encoding

        soup = BeautifulSoup(res.text, "html.parser")

        rows = []

        for tr in soup.find_all("tr"):

            cols = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]

            if len(cols) > 3:

                rows.append(cols)

        if len(rows) > 1:

            header = rows[0]

            data_rows = []

            for row in rows[1:]:

                if len(row) < len(header):

                    row = row + [""] * (len(header) - len(row))

                elif len(row) > len(header):

                    row = row[:len(header)]

                data_rows.append(row)

            df = pd.DataFrame(data_rows, columns=header)

            return df, None

        return pd.DataFrame(), "テーブルが見つかりませんでした"

    except Exception as e:

        return pd.DataFrame(), str(e)

 

import streamlit as st

import pandas as pd

import requests

from bs4 import BeautifulSoup

import urllib3

 

def fetch_result_table(url):

    """レース結果ページから着順・馬名を取得"""

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}

        res = requests.get(url, headers=headers, verify=False, timeout=10)

        res.encoding = res.apparent_encoding

        soup = BeautifulSoup(res.text, "html.parser")

        rows = []

        for tr in soup.find_all("tr"):

            cols = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]

            if len(cols) > 3:

                rows.append(cols)

        if len(rows) > 1:

            header = rows[0]

            # 着順列名候補

            rank_candidates = ["着順", "着", "順位"]

            name_candidates = ["馬名", "馬"]

            rank_col = next((c for c in rank_candidates if c in header), None)

            name_col = next((c for c in name_candidates if c in header), None)

            if rank_col and name_col:

                # データ行の長さをheaderに合わせて調整

                data_rows = []

                for row in rows[1:]:

                    if len(row) < len(header):

                        row = row + [""] * (len(header) - len(row))

                    elif len(row) > len(header):

                        row = row[:len(header)]

                    data_rows.append(row)

                df = pd.DataFrame(data_rows, columns=header)

                # 列名を統一

                df = df.rename(columns={rank_col: "着順", name_col: "馬名"})

                # デバッグ表示抑制

                try:

                    return df[["馬名", "着順"]]

                except KeyError as e:

                    # st.write(f"[デバッグ] KeyError: {e}")  # デバッグ表示抑制

                    return pd.DataFrame()

            else:

                # st.write(f"[デバッグ] header: {header}")  # デバッグ表示抑制

                return pd.DataFrame()

        return pd.DataFrame()

    except Exception as e:

        print(f"[DEBUG] fetch_result_table error: {e}")

        return pd.DataFrame()

def main():

    st.subheader("レース結果（着順）データ追加")

    result_url = st.text_input("レース結果ページのURLを入力してください（例: syutsuba.html→raceresult.html）", "")

    if st.button("着順データを取得して出馬表とマージ・保存"):

        # --- デバッグ表示削除 ---

        # --- 既存の着順抽出ロジックも実行 ---

        df_result = fetch_result_table(result_url)

        # デバッグ表示抑制

        if df_result.empty:

            st.warning("着順データの取得に失敗しました")

        else:

            # st.success("着順データ取得成功！出馬表とマージします")  # 表示抑制

            # セッションから出馬表データを取得

            if 'df' in st.session_state and not st.session_state['df'].empty:

                try:

                    df_merged = pd.merge(st.session_state['df'], df_result, on="馬名", how="left")

                    # local_train.csvに追記

                    import os

                    train_path = "local_train.csv"

                    # 既存ファイルがあればカラム統一して追記（重複排除も追加）

                    if os.path.exists(train_path):

                        try:

                            import unicodedata

                            df_old = pd.read_csv(train_path)

                            all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_merged.columns)))

                            df_old2 = df_old.reindex(columns=all_cols)

                            df_new2 = df_merged.reindex(columns=all_cols)

                            df_all = pd.concat([df_old2, df_new2], ignore_index=True)

                            # --- 重複排除（馬名・調教師・騎手・枠で最新のみ）---

                            key_cols = ["馬名", "調教師", "騎手", "枠"]

                            # st.write(f"[デバッグ] 保存前: {len(df_all)}件（グループ化キー: {key_cols}）")  # デバッグ表示抑制

                            # グループ化キーをstr型・strip・全角半角正規化・記号除去

                            if all(c in df_all.columns for c in key_cols):

                                for c in key_cols:

                                    df_all[c] = df_all[c].astype(str).str.strip()

                                    df_all[c] = df_all[c].apply(lambda x: unicodedata.normalize('NFKC', x))

                                    df_all[c] = df_all[c].replace({r'[\s　]+': '', r'[…注－―ー・,，．。\.\(\)\[\]{}<>「」『』【】]': ''}, regex=True)

                                df_all = df_all.groupby(key_cols, as_index=False).first().reset_index(drop=True)

                                for col in df_all.columns:

                                    df_all[col] = df_all[col].astype(str).str.strip()

                                df_all = df_all.drop_duplicates()

                            else:

                                st.warning(f"グループ化キーが存在しません: {key_cols}")

                            # st.write(f"[デバッグ] グループ化後: {len(df_all)}件（完全一致重複も除去済み）")  # デバッグ表示抑制

                            df_all.to_csv(train_path, index=False, encoding="utf-8-sig")

                        except Exception:

                            df_merged.to_csv(train_path, index=False, encoding="utf-8-sig")

                            st.warning("既存の学習データが空または壊れていたため新規保存しました")

                    else:

                        df_merged.to_csv(train_path, index=False, encoding="utf-8-sig")

                    st.success("着順付きデータをlocal_train.csvに保存しました（重複排除済み）")

                except Exception as e:

                    st.warning(f"マージ・保存時エラー: {e}")

            else:

                st.warning("先に出馬表データを取得してください")

    st.title("競馬ラボ出馬表 予想アプリ")

    url = st.text_input("出馬表のURLを入力してください", "")

    if url:

        df, err = fetch_table(url)

        if err:

            st.error(f"データ取得エラー: {err}")

        elif not df.empty:

            st.session_state['df'] = df.copy()  # セッションに保存
            st.write("馬名一覧",df_ai["馬名"].tolist())
            st.write("馬名欠損数",df_ai["馬名"].isnull().sum())
            # --- 出馬表データの表示は省略（AI予想のみ表示） ---

            import os

            from sklearn.ensemble import RandomForestClassifier

            from sklearn.model_selection import train_test_split

            from sklearn.preprocessing import LabelEncoder

            train_path = "local_train.csv"

            df_ai = df.copy()

            # --- 特徴量自動選択・強化 ---

            # --- 主要数値列の数値抽出 ---

            for col in ["馬体重", "斤量"]:

                if col in df_ai.columns:

                    # 例: "454(+2)" → 454.0

                    df_ai[col] = df_ai[col].astype(str).str.extract(r'(\d+)').astype(float)

            for col in ["単勝", "人気"]:

                if col in df_ai.columns:

                    df_ai[col] = pd.to_numeric(df_ai[col], errors='coerce')

 

            # --- 組み合わせ特徴量のさらなる自動化（積・比・差分・和・絶対値・対数） ---

            import numpy as np

            num_cols_comb = []

            for col in df_ai.columns:

                vals = pd.to_numeric(df_ai[col], errors='coerce')

                if vals.notnull().sum() > 0 and vals.nunique(dropna=True) > 1 and vals.notnull().sum() > 0:

                    num_cols_comb.append(col)

            from itertools import combinations

            for col1, col2 in combinations(num_cols_comb, 2):

                try:

                    v1 = pd.to_numeric(df_ai[col1], errors='coerce')

                    v2 = pd.to_numeric(df_ai[col2], errors='coerce')

                    df_ai[f"{col1}_x_{col2}_mul"] = v1 * v2

                    df_ai[f"{col1}_div_{col2}_div"] = v1 / (v2 + 1e-8)

                    df_ai[f"{col1}_sub_{col2}_sub"] = v1 - v2

                    df_ai[f"{col1}_add_{col2}_add"] = v1 + v2

                    df_ai[f"{col1}_abs_{col2}_abs"] = (v1 - v2).abs()

                except Exception:

                    pass

            # 対数変換特徴量

            for col in num_cols_comb:

                try:

                    v = pd.to_numeric(df_ai[col], errors='coerce')

                    df_ai[f"{col}_log"] = np.log1p(v.clip(lower=0))

                except Exception:

                    pass

 

            # --- カテゴリ×数値の集約特徴量（騎手・調教師ごとの平均着順・平均単勝） ---

            for cat in [c for c in ["騎手", "調教師"] if c in df_ai.columns]:

                for num in [c for c in ["着順", "単勝"] if c in df_ai.columns]:

                    try:

                        grp = df_ai.groupby(cat)[num].transform('mean')

                        df_ai[f"{cat}_{num}_mean"] = grp

                    except Exception:

                        pass

 

            # --- 欠損値の高度な自動補完（平均/中央値/最頻値） ---

            for col in df_ai.columns:

                if df_ai[col].isnull().any():

                    if df_ai[col].dtype.kind in 'biufc':

                        df_ai[col] = df_ai[col].fillna(df_ai[col].mean())

                        if df_ai[col].isnull().any():

                            df_ai[col] = df_ai[col].fillna(df_ai[col].median())

                    else:

                        df_ai[col] = df_ai[col].fillna(df_ai[col].mode().iloc[0] if not df_ai[col].mode().empty else "")

            # 記号や…をNaNに変換

            df_ai = df_ai.replace({r'^[^\d\w]+$': None, '…': None, '－': None, '―': None, 'ー': None, '': None}, regex=True)

            # 性齢分解

            if "性齢" in df_ai.columns:

                df_ai["性"] = df_ai["性齢"].astype(str).str.extract(r'(牡|牝|セ)')

                df_ai["年齢"] = df_ai["性齢"].astype(str).str.extract(r'(\d+)').astype(float)

            # カテゴリ変数候補

            cat_cols = [col for col in ["騎手", "調教師", "性"] if col in df_ai.columns]

            # 数値列候補（全NaN/定数列は除外）

            num_cols = []

            for col in df_ai.columns:

                vals = pd.to_numeric(df_ai[col], errors='coerce')

                if vals.notnull().sum() > 0 and vals.nunique(dropna=True) > 1 and vals.notnull().sum() > 0:

                    num_cols.append(col)

            # 代表的な特徴量候補

            feature_candidates = ["単勝", "枠", "馬体重", "人気", "年齢", "斤量"]

            features = [col for col in feature_candidates if col in df_ai.columns]

            # 自動判定で数値列も追加

            for col in num_cols:

                if col not in features:

                    features.append(col)

            # カテゴリ変数も追加

            for col in cat_cols:

                if col not in features:

                    features.append(col)

            # --- カテゴリ変数エンコーディング ---

            encoders = {}

            for col in cat_cols:

                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()

                df_ai[col+"_enc"] = le.fit_transform(df_ai[col].astype(str))

                encoders[col] = le

            # 特徴量リストをエンコーディング後の列に置換

            features_final = []

            for col in features:

                if col in cat_cols:

                    features_final.append(col+"_enc")

                else:

                    features_final.append(col)

            # 全てNaN列・定数列は除外

            features_final = [col for col in features_final if df_ai[col].nunique(dropna=True) > 1 and df_ai[col].notnull().sum() > 0]

            # --- 追加: 相関係数による冗長特徴量除去 ---

            if len(features_final) > 1:

                corr_matrix = df_ai[features_final].corr().abs()

                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

                # st.info(f"[特徴量選択] 相関係数0.95超の冗長特徴量を自動除外: {to_drop}")  # 表示抑制

                features_final = [col for col in features_final if col not in to_drop]

            # --- 追加: L1正則化（Lasso）による自動特徴量選択 ---

            if len(features_final) > 1:

                from sklearn.linear_model import LogisticRegression

                try:

                    X_l1 = df_ai[features_final].apply(pd.to_numeric, errors='coerce').fillna(0)

                    y_l1 = None

                    # local_train.csvがあれば着順1着/それ以外で学習

                    if os.path.exists(train_path):

                        df_train_l1 = pd.read_csv(train_path)

                        if "着順" in df_train_l1.columns:

                            # local_train.csvに存在する特徴量だけに絞る

                            l1_cols = [f for f in features_final if f in df_train_l1.columns]

                            if l1_cols:

                                y_l1 = (pd.to_numeric(df_train_l1["着順"], errors='coerce') == 1).astype(int)

                                X_l1 = df_train_l1[l1_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

                            else:

                                y_l1 = None

                    if y_l1 is not None and y_l1.nunique() > 1:

                        l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)

                        l1.fit(X_l1, y_l1)

                        coef_mask = l1.coef_[0] != 0

                        l1_selected = [f for f, use in zip(X_l1.columns, coef_mask) if use]

                        if len(l1_selected) < len(features_final) and len(l1_selected) > 0:

                            # st.info(f"[特徴量選択] L1正則化で自動選択: {l1_selected}")  # 表示抑制

                            features_final = l1_selected

                except Exception as e:

                    st.write(f"[L1特徴量選択エラー] {e}")

            if not features_final:

                st.warning("有効な特徴量がありません。データに記号や定数列・全NaN列が多い可能性があります。")

            # --- 目的変数・学習データ前処理も同様に強化 ---

            # 目的変数

            model = None

            model_name = "RandomForest"

            import numpy as np

            if os.path.exists(train_path):

                try:

                    df_train = pd.read_csv(train_path)

                    # --- 主要数値列の数値抽出 ---

                    for col in ["馬体重", "斤量"]:

                        if col in df_train.columns:

                            df_train[col] = df_train[col].astype(str).str.extract(r'(\d+)').astype(float)

                    for col in ["単勝", "人気"]:

                        if col in df_train.columns:

                            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

 

                    # --- 組み合わせ特徴量のさらなる自動化（積・比・差分・和・絶対値・対数） ---

                    import numpy as np

                    num_cols_comb = []

                    for col in df_train.columns:

                        vals = pd.to_numeric(df_train[col], errors='coerce')

                        if vals.notnull().sum() > 0 and vals.nunique(dropna=True) > 1 and vals.notnull().sum() > 0:

                            num_cols_comb.append(col)

                    from itertools import combinations

                    for col1, col2 in combinations(num_cols_comb, 2):

                        try:

                            v1 = pd.to_numeric(df_train[col1], errors='coerce')

                            v2 = pd.to_numeric(df_train[col2], errors='coerce')

                            df_train[f"{col1}_x_{col2}_mul"] = v1 * v2

                            df_train[f"{col1}_div_{col2}_div"] = v1 / (v2 + 1e-8)

                            df_train[f"{col1}_sub_{col2}_sub"] = v1 - v2

                            df_train[f"{col1}_add_{col2}_add"] = v1 + v2

                            df_train[f"{col1}_abs_{col2}_abs"] = (v1 - v2).abs()

                        except Exception:

                            pass

                    # 対数変換特徴量

                    for col in num_cols_comb:

                        try:

                            v = pd.to_numeric(df_train[col], errors='coerce')

                            df_train[f"{col}_log"] = np.log1p(v.clip(lower=0))

                        except Exception:

                            pass

 

                    # --- カテゴリ×数値の集約特徴量（騎手・調教師ごとの平均着順・平均単勝） ---

                    for cat in [c for c in ["騎手", "調教師"] if c in df_train.columns]:

                        for num in [c for c in ["着順", "単勝"] if c in df_train.columns]:

                            try:

                                grp = df_train.groupby(cat)[num].transform('mean')

                                df_train[f"{cat}_{num}_mean"] = grp

                            except Exception:

                                pass

 

                    # --- 欠損値の高度な自動補完（平均/中央値/最頻値） ---

                    for col in df_train.columns:

                        if df_train[col].isnull().any():

                            if df_train[col].dtype.kind in 'biufc':

                                df_train[col] = df_train[col].fillna(df_train[col].mean())

                                if df_train[col].isnull().any():

                                    df_train[col] = df_train[col].fillna(df_train[col].median())

                            else:

                                df_train[col] = df_train[col].fillna(df_train[col].mode().iloc[0] if not df_train[col].mode().empty else "")

                    # 重複排除（馬名＋調教師＋騎手＋枠などで最新のみ）

                    key_cols = [c for c in ["馬名", "調教師", "騎手", "枠"] if c in df_train.columns]

                    if key_cols:

                        df_train = df_train.drop_duplicates(subset=key_cols, keep="last")

                    # 性齢分解・カテゴリ変数エンコーディング

                    if "性齢" in df_train.columns:

                        df_train["性"] = df_train["性齢"].astype(str).str.extract(r'(牡|牝|セ)')

                        df_train["年齢"] = df_train["性齢"].astype(str).str.extract(r'(\d+)').astype(float)

                    for col in cat_cols:

                        if col in df_train.columns:

                            le = LabelEncoder()

                            df_train[col+"_enc"] = le.fit_transform(df_train[col].astype(str))

                    # 着順列を厳密に数値型に変換し、空欄やNaNを除外

                    if "着順" in df_train.columns:

                        df_train = df_train.copy()

                        df_train["着順"] = pd.to_numeric(df_train["着順"], errors='coerce')

                        df_train = df_train[df_train["着順"].notnull()]

                    # 特徴量・目的変数の欠損除外

                    valid_cols = features_final + ["着順"]

                    valid_cols = [c for c in valid_cols if c in df_train.columns]

                    df_train = df_train.dropna(subset=valid_cols)

                    # 必要なカラムが揃っているか

                    if ("着順" in df_train.columns) and all(f in df_train.columns or f.replace("_enc","") in df_train.columns for f in features_final):

                        X = df_train[features_final].apply(pd.to_numeric, errors='coerce').fillna(0)

                        y = df_train["着順"].astype(int)

                        y_bin = (y == 1).astype(int)

                        use_lgb = False

                        try:

                            import lightgbm as lgb

                            use_lgb = True

                        except ImportError:

                            pass

                        from sklearn.model_selection import cross_val_score, StratifiedKFold

                        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

                        # --- fold数自動調整の最適化: 各foldに1着が必ず入る最大fold数を自動計算 ---

                        n_pos = int((y_bin == 1).sum())

                        n_total = len(y_bin)

                        max_folds = min(10, n_pos, n_total)  # 10fold以上は推奨しない

                        # fold数候補: 2〜max_folds で、全foldに1着が最低1件入るもの

                        best_folds = 2

                        for f in range(max_folds, 1, -1):

                            if n_pos // f > 0 and n_total // f >= 5:  # 1foldあたり最低5件

                                best_folds = f

                                break

                        n_folds = best_folds

                        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

                        eval_results = {}

                        if use_lgb:

                            # --- 追加: 特徴量・目的変数の健全性チェック ---

                            if X.shape[1] == 0 or (X.nunique() <= 1).any() or X.std().sum() == 0:

                                st.warning("有効な特徴量がありません。全て定数列またはNaN列の可能性があります。AI学習をスキップします。")

                                model = None

                            elif y_bin.nunique() < 2:

                                st.warning("目的変数（1着/それ以外）のクラスが1つしかありません。AI学習をスキップします。")

                                model = None

                            else:

                                # --- クラス重み自動調整 ---

                                pos = (y_bin == 1).sum()

                                neg = (y_bin == 0).sum()

                                scale_pos_weight = float(neg) / max(float(pos), 1)

                                # OptunaによるLightGBM自動最適化

                                use_optuna = False

                                try:

                                    import optuna

                                    use_optuna = True

                                except ImportError:

                                    pass

                                if use_optuna:

                                    # st.info("OptunaによるLightGBM自動最適化を実行中...")  # 表示抑制

                                    def objective(trial):

                                        param = {

                                            'objective': 'binary',

                                            'metric': 'binary_logloss',

                                            'verbosity': -1,

                                            'boosting_type': 'gbdt',

                                            'seed': 42,

                                            'scale_pos_weight': scale_pos_weight,

                                            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),

                                            'num_leaves': trial.suggest_int('num_leaves', 8, 128),

                                            'max_depth': trial.suggest_int('max_depth', 3, 10),

                                            'min_child_samples': trial.suggest_int('min_child_samples', 2, 30),

                                            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),

                                            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),

                                            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),

                                            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),

                                        }

                                        from lightgbm import LGBMClassifier

                                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                                        model = LGBMClassifier(**param, n_estimators=100, random_state=42)

                                        scores = cross_val_score(model, X, y_bin, cv=cv, scoring='roc_auc')

                                        return scores.mean()

                                    study = optuna.create_study(direction='maximize')

                                    with st.spinner('Optunaでパラメータ探索中...（数十秒かかる場合があります）'):

                                        study.optimize(objective, n_trials=20, show_progress_bar=False)

                                    best_params = study.best_params

                                    # st.success(f"Optuna最適化完了！最良パラメータ: {best_params}")  # 表示抑制

                                    # 最適パラメータでLightGBM学習

                                    import lightgbm as lgb

                                    lgb_train = lgb.Dataset(X, y_bin)

                                    params = best_params.copy()

                                    params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'seed': 42, 'scale_pos_weight': scale_pos_weight})

                                    gbm = lgb.train(params, lgb_train, num_boost_round=100)

                                    model = gbm

                                    model_name = "LightGBM(Optuna)"

                                    # st.info(f"LightGBM(Optuna)で自動学習済み（着順1着分類, {len(df_train)}行, 特徴量: {features_final}）")  # 表示抑制

                                else:

                                    model_name = "LightGBM"

                                    lgb_train = lgb.Dataset(X, y_bin)

                                    params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42, "scale_pos_weight": scale_pos_weight}

                                    gbm = lgb.train(params, lgb_train, num_boost_round=100)

                                    model = gbm

                                    # st.info(f"LightGBMで自動学習済み（着順1着分類, {len(df_train)}行, 特徴量: {features_final}）")  # 表示抑制

                                # --- 特徴量選択の自動化: 重要度0の特徴量を除外し再学習 ---

                                try:

                                    importances = model.feature_importance(importance_type="gain")

                                    feature_names = list(X.columns)

                                    selected_features = [f for f, imp in zip(feature_names, importances) if imp > 0]

                                    if len(selected_features) < len(feature_names) and len(selected_features) > 0:

                                        # st.info(f"[特徴量選択] 重要度0の特徴量を除外し再学習: {selected_features}")  # 表示抑制

                                        X_sel = X[selected_features]

                                        # 再学習

                                        lgb_train_sel = lgb.Dataset(X_sel, y_bin)

                                        gbm_sel = lgb.train(params, lgb_train_sel, num_boost_round=100)

                                        model = gbm_sel

                                        X = X_sel

                                        features_final = selected_features

                                except Exception as e:

                                    st.write(f"[特徴量選択エラー] {e}")

                                # クロスバリデーション評価

                                from lightgbm import LGBMClassifier

                                lgbm_cv = LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)

                                aucs = cross_val_score(lgbm_cv, X, y_bin, cv=cv, scoring='roc_auc')

                                f1s = cross_val_score(lgbm_cv, X, y_bin, cv=cv, scoring='f1')

                                accs = cross_val_score(lgbm_cv, X, y_bin, cv=cv, scoring='accuracy')

                                eval_results = {'AUC': aucs.mean(), 'F1': f1s.mean(), 'Accuracy': accs.mean()}

                        else:

                            # RandomForestパラメータ最適化（簡易GridSearchCV）

                            from sklearn.model_selection import GridSearchCV

                            param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5, None]}

                            # --- クラス重み自動調整 ---

                            pos = (y_bin == 1).sum()

                            neg = (y_bin == 0).sum()

                            class_weight = {0: 1.0, 1: max(float(neg) / max(float(pos), 1), 1.0)}

                            rf = RandomForestClassifier(random_state=42, class_weight=class_weight)

                            grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, error_score="raise")

                            grid.fit(X, y_bin)

                            model = grid.best_estimator_

                            # st.info(f"RandomForestで自動学習済み（最適パラメータ: {grid.best_params_}, {len(df_train)}行, 特徴量: {features_final}、class_weight: {class_weight}）")  # 表示抑制

                            # クロスバリデーション評価

                            aucs = cross_val_score(model, X, y_bin, cv=cv, scoring='roc_auc')

                            f1s = cross_val_score(model, X, y_bin, cv=cv, scoring='f1')

                            accs = cross_val_score(model, X, y_bin, cv=cv, scoring='accuracy')

                            eval_results = {'AUC': aucs.mean(), 'F1': f1s.mean(), 'Accuracy': accs.mean()}

                        # --- 予測・評価の多様化: 混同行列 ---

                        # --- 混同行列・評価指標などの表示は省略 ---

                        # --- 特徴量重要度自動選択: Permutation Importance（可視化なし） ---

                        # --- Permutation Importance等のデバッグ・表示も省略 ---

                        # --- 特徴量重要度表示（従来通り） ---

                        # --- 特徴量重要度等の表示も省略 ---

                except Exception as e:

                    st.warning(f"AI自動学習エラー: {e}")

            # --- 予想 ---

            if model is not None:

                X_pred = df_ai[features_final].apply(pd.to_numeric, errors='coerce').fillna(0)

                is_lgb = False

                try:

                    import lightgbm as lgb

                    if isinstance(model, lgb.Booster):

                        is_lgb = True

                except Exception:

                    pass

                if is_lgb:

                    y_pred_prob = model.predict(X_pred)

                    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:

                        y_pred_prob = y_pred_prob[:,1]

                else:

                    if hasattr(model, 'classes_') and len(model.classes_) < 2:

                        y_pred_prob = model.predict(X_pred)

                    else:

                        y_pred_prob = model.predict_proba(X_pred)[:,1]

                df_ai["AI_1着確率"] = y_pred_prob

                # シンプルな印付け（確率順で◎○▲△）

                df_ai["AI印"] = ""

                n = len(df_ai)

                marks = ["◎", "○", "▲"] + ["△"] * max(0, n-3)

                order = df_ai["AI_1着確率"].values.argsort()[::-1]

                for idx, mark in zip(order, marks):

                    df_ai.at[idx, "AI印"] = mark

                # シンプルなテーブルのみ表示

                pred_df = df_ai[["馬名", "AI印", "AI_1着確率"]].copy() if "馬名" in df_ai.columns else df_ai[["AI印", "AI_1着確率"]].copy()

                pred_df = pred_df.sort_values("AI_1着確率", ascending=False).reset_index(drop=True)

                st.subheader("AI予想（シンプル表示）")

                st.dataframe(pred_df.style.format({"AI_1着確率": "{:.3f}"}))

            else:

                st.dataframe(df_ai)

                st.warning("AIモデル学習用データが不足しています。着順付きデータを蓄積してください。")

            # --- CSVダウンロードボタン ---

            csv = df_ai.to_csv(index=False, encoding="utf-8-sig")

            st.download_button("この出馬表データをCSVで保存", csv, file_name="syutsuba.csv", mime="text/csv")

            # --- 出馬表取得時はlocal_train.csvへ保存しない ---

            # 保存は着順マージ時のみ行う

        else:

            st.write("データ取得に失敗しました")

 

if __name__ == "__main__":

    main()
