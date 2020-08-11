from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline, Transformer
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):

    word_tokenizer = Tokenizer(inputCol=input_descript_col, outputCol="words")

    count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)

    label_maker = StringIndexer(inputCol=input_category_col, outputCol=output_label_col)

    class Selector(Transformer):
        def __init__(self, outputCols=['id', 'features', 'label']):
            self.outputCols = outputCols

        def _transform(self, df: DataFrame) -> DataFrame:
            return df.select(*self.outputCols)

    selector = Selector(outputCols=['id', 'features', 'label'])
    pipeline = Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])

    return pipeline

def joint_pred(df):
    df = df.withColumn("joint_pred_0", -1 * F.col("id"))
    df = df.withColumn("joint_pred_1", -1 * F.col("id"))
    df = df.withColumn("joint_pred_2", -1 * F.col("id"))

    df = df.withColumn("joint_pred_0", F.when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 0), 0).otherwise(
        F.col("joint_pred_0")))

    df = df.withColumn("joint_pred_0", F.when((F.col("nb_pred_0") == 0) & (F.col("svm_pred_0") == 1), 1).otherwise(
        F.col("joint_pred_0")))

    df = df.withColumn("joint_pred_0", F.when((F.col("nb_pred_0") == 1) & (F.col("svm_pred_0") == 0), 2).otherwise(
        F.col("joint_pred_0")))

    df = df.withColumn("joint_pred_0", F.when((F.col("nb_pred_0") == 1) & (F.col("svm_pred_0") == 1), 3).otherwise(
        F.col("joint_pred_0")))

    #######################################################

    df = df.withColumn("joint_pred_1", F.when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 0), 0).otherwise(
        F.col("joint_pred_1")))

    df = df.withColumn("joint_pred_1", F.when((F.col("nb_pred_1") == 0) & (F.col("svm_pred_1") == 1), 1).otherwise(
        F.col("joint_pred_1")))

    df = df.withColumn("joint_pred_1", F.when((F.col("nb_pred_1") == 1) & (F.col("svm_pred_1") == 0), 2).otherwise(
        F.col("joint_pred_1")))

    df = df.withColumn("joint_pred_1", F.when((F.col("nb_pred_1") == 1) & (F.col("svm_pred_1") == 1), 3).otherwise(
        F.col("joint_pred_1")))

    #######################################################

    df = df.withColumn("joint_pred_2", F.when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 0), 0).otherwise(
        F.col("joint_pred_2")))

    df = df.withColumn("joint_pred_2", F.when((F.col("nb_pred_2") == 0) & (F.col("svm_pred_2") == 1), 1).otherwise(
        F.col("joint_pred_2")))

    df = df.withColumn("joint_pred_2", F.when((F.col("nb_pred_2") == 1) & (F.col("svm_pred_2") == 0), 2).otherwise(
        F.col("joint_pred_2")))

    df = df.withColumn("joint_pred_2", F.when((F.col("nb_pred_2") == 1) & (F.col("svm_pred_2") == 1), 3).otherwise(
        F.col("joint_pred_2")))

    return df

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    for i in range(5):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()

        nb_model_0 = nb_0.fit(c_train)
        nb_pred_0 = nb_model_0.transform(c_test)

        nb_model_1 = nb_1.fit(c_train)
        nb_pred_1 = nb_model_1.transform(c_test)

        nb_model_2 = nb_2.fit(c_train)
        nb_pred_2 = nb_model_2.transform(c_test)

        svm_model_0 = svm_0.fit(c_train)
        svm_pred_0 = svm_model_0.transform(c_test)

        svm_model_1 = svm_1.fit(c_train)
        svm_pred_1 = svm_model_1.transform(c_test)

        svm_model_2 = svm_2.fit(c_train)
        svm_pred_2 = svm_model_2.transform(c_test)

        nb_pred_0 = nb_pred_0.join(nb_pred_1, ['id']).select(nb_pred_0["*"],nb_pred_1['nb_pred_1'])
        nb_pred_0 = nb_pred_0.join(nb_pred_2, ['id']).select(nb_pred_0["*"],nb_pred_2['nb_pred_2'])
        nb_pred_0 = nb_pred_0.join(svm_pred_0, ['id']).select(nb_pred_0["*"],svm_pred_0['svm_pred_0'])
        nb_pred_0 = nb_pred_0.join(svm_pred_1, ['id']).select(nb_pred_0["*"],svm_pred_1['svm_pred_1'])
        df = nb_pred_0.join(svm_pred_2, ['id']).select(nb_pred_0["*"], svm_pred_2['svm_pred_2'])

        if i==0:
            df0 = joint_pred(df)

        elif i==1:
            df1 = joint_pred(df)

        elif i==2:
            df2 = joint_pred(df)

        elif i==3:
            df3 = joint_pred(df)

        elif i==4:
            df4 = joint_pred(df)

    df_tot=df0.unionAll(df1).unionAll(df2).unionAll(df3).unionAll(df4)
    nb_drop_0 = ['nb_raw_0', 'nb_prob_0']
    df_tot = df_tot.drop(*nb_drop_0)

    return df_tot

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    test_set=base_features_pipeline_model.transform(test_df)
    base_features=gen_base_pred_pipeline_model.transform(test_set)
    joint_features=joint_pred(base_features)
    test_meta_features=gen_meta_feature_pipeline_model.transform(joint_features)
    pred=meta_classifier.transform(test_meta_features)

    drop_col=['features', 'nb_raw_0', 'nb_prob_0', 'nb_pred_0', 'nb_raw_1', 'nb_prob_1', 'nb_pred_1', 'nb_raw_2',
     'nb_prob_2', 'nb_pred_2', 'svm_raw_0', 'svm_pred_0', 'svm_raw_1', 'svm_pred_1', 'svm_raw_2', 'svm_pred_2',
     'joint_pred_0', 'joint_pred_1', 'joint_pred_2', 'vec4', 'vec7', 'vec0', 'vec1', 'vec6', 'vec2', 'vec5', 'vec3',
     'vec8', 'meta_features', 'rawPrediction', 'probability']

    pred_f = pred.drop(*drop_col)

    return pred_f