# flow


# for 
shot = 5
seeds = list(range(100))

model = model()
result = []
for seed in seeds:
    train_dataset, val_dataset, test_dataset = build_dataset()
    train_dataset = build_fewshot_dataset(train_dataset, shots, seed)
    test_dataset = build_test(val_dataset, test_dataset)

        # CV with k-shot dataset
    if model == "lgbm":
        estimator = lgb.LGBMClassifier(class_weight='balanced', num_threads=1, random_state=seed)
    elif model == "linearl1":
        estimator = Lasso()
    folds = 5
    inner_cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv, scoring=metric, n_jobs=40, verbose=0)
    clf.fit(X_train, y_train)
    
    clf.predict(X_test)
    acc = ...
    result.append(acc)