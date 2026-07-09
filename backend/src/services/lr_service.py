from services.model_loader import (tfidf, lr_model)
from schemas.response import LRResult, ProbabilityOutput, LRExplanations, ExplanationWord

def lr_predict(texts): 
    X = tfidf.transform([texts]) 
    # 0 = Fake, 1 = Real
    lr_pred = lr_model.predict(X)[0]
    # [fake_prob, real_prob] 
    lr_prob = lr_model.predict_proba(X)[0]
    # Get Local LR model contributions 
    features_name = tfidf.get_feature_names_out()
    weights = lr_model.coef_[0]
    active_indices = X.nonzero()[1]
    contributions = []
    
    for idx in active_indices: 
        feature = features_name[idx] 
        tfidf_value = X[0, idx] 
        weight = weights[idx]
        contribution = tfidf_value * weight
        contributions.append(ExplanationWord(word=str(feature), score=float(contribution))) 
        
    contributions = sorted(contributions, key=lambda x: abs(x.score), reverse=True)
    # Separate contributions into fake and real 
    push_fake = [x for x in contributions if x.score < 0] 
    push_real = [x for x in contributions if x.score > 0] 
    
    push_fake = sorted(push_fake, key=lambda x: x.score) 
    push_real = sorted(push_real, key=lambda x: x.score, reverse=True) 
    # Global explanation 
    # word_weights = list(zip(features_name, weights)) 
    # model_weights = sorted(word_weights, key=lambda x: x[1], reverse=True) 
    
    return LRResult(
        prediction="Real" if lr_pred == 1 else "Fake",
        prediction_id=lr_pred,
        probabilities=ProbabilityOutput(
            fake=float(lr_prob[0]),
            real=float(lr_prob[1])
        ),
        explanations=LRExplanations(
            push_fake=push_fake[:10],
            push_real=push_real[:10]
        )
    )