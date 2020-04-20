pretrain_masked_model1 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0.8759408295591542
    GraphPositionPredictAccuracy: 0.9751876047386078
    ''',
    'all_result': 0.9532761664078401,
    'masked_result': 0.9533248637087307,

    'config_name': 'pretrain_masked_model1',
    'used_model': 'pretrain_masked_model1.pkl41',

    'train_type': 'both',
    'task': 'position',
    'loss_range': 'all',
}

pretrain_masked_model2 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0
    GraphPositionPredictAccuracy: 0.9819772210748825
    ''',
    'all_result': 0.9683818026907863,
    'masked_result': 0.9684095471926986,

    'config_name': 'pretrain_masked_model2',
    'used_model': 'pretrain_masked_model2.pkl13',

    'train_type': 'only_disc',
    'task': 'position',
    'loss_range': 'all',
}

pretrain_masked_model4 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0.8761054740305216
    GraphPositionPredictAccuracy: 0.9708689922201765
    ''',
    'all_result': 0.9316981544735843,
    'masked_result': 0.9317390884821974,

    'config_name': 'pretrain_masked_model4_gene20',
    'used_model': ['pretrain_masked_model3.pkl20', 'pretrain_masked_model4_gene20.pkl38'],

    'train_type': 'gene+disc',
    'task': 'position',
    'loss_range': 'all',
}

pretrain_masked_model5 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0
    GraphPositionPredictAccuracy: 0.7506716818914562
    ''',
    'all_result': 0.4134765637559724,
    'masked_result': 0.41340677927949654,

    'config_name': 'pretrain_masked_model5',
    'used_model': 'pretrain_masked_model5.pkl35',

    'train_type': 'bert',
    'task': 'word',
    'loss_range': 'only_masked',
}

pretrain_masked_model6 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0.8641472552532986
    GraphPositionPredictAccuracy: 0.9718688994876775
    ''',
    'all_result': 0.9492621706463565,
    'masked_result': 0.9058748341584959,

    'config_name': 'pretrain_masked_model6.pkl33',
    'used_model': '',

    'train_type': 'both',
    'task': 'word',
    'loss_range': 'all',
}

pretrain_masked_model7 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0.8847989315438866
    GraphPositionPredictAccuracy: 0.9380588628108866
    ''',
    'all_result': 0.9154845784519602,
    'masked_result': 0.8948560702901247,

    'config_name': 'pretrain_masked_model7',
    'used_model': 'pretrain_masked_model7.pkl45',

    'train_type': 'both',
    'task': 'word',
    'loss_range': 'only_masked',
}

pretrain_masked_model8 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0.8601325711307634
    GraphPositionPredictAccuracy: 0.9347241901597694
    ''',
    'all_result': 0.847197522487032,
    'masked_result': 0.8873798038009518,

    'config_name': 'pretrain_masked_model8',
    'used_model': 'pretrain_masked_model8.pkl30',

    'train_type': 'both',
    'task': 'position',
    'loss_range': 'only_masked',
}

pretrain_masked_model9 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0.8761054740305216
    GraphPositionPredictAccuracy: 0.9243014256619144
    ''',
    'all_result': 0.895862135485323,
    'masked_result': 0.8744489294063402,

    'config_name': 'pretrain_masked_model9',
    'used_model': ['pretrain_masked_model3.pkl20', 'pretrain_masked_model9_gene20.pkl31'],

    'train_type': 'gene+disc',
    'task': 'word',
    'loss_range': 'only_masked',
}

pretrain_masked_model10 = {
    'result': '''
    MaskedLanguageModelTokenAccuracy: 0
    GraphPositionPredictAccuracy: 0.8731311683658513
    ''',
    'all_result': 0.9600169812048936,
    'masked_result': 0.926921862731607,

    'config_name': 'pretrain_masked_model10',
    'used_model': 'pretrain_masked_model10.pkl32',

    'train_type': 'only_disc',
    'task': 'word',
    'loss_range': 'only_masked',
}

