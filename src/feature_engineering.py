def create_features(df):

    # Engagement Score
    df['EngagementScore'] = (
        df['ProductRelated'] * df['PageValues']
    ) / (df['BounceRates'] + 0.01)

    # Activity Score
    df['ActivityScore'] = (
        df['Administrative_Duration'] +
        df['Informational_Duration'] +
        df['ProductRelated_Duration']
    )

    # Risk Score
    df['RiskScore'] = df['BounceRates'] + df['ExitRates']

    # Returning visitor
    df['IsReturning'] = (df['VisitorType'] == 'Returning_Visitor').astype(int)

    # Weekend flag
    df['Weekend'] = df['Weekend'].astype(int)

    return df