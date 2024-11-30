import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Expanded dataset
data = {
    'Disease': [
        'Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'Anemia', 
        'Obesity', 'Osteoporosis', 'Allergies', 'Arthritis', 'Cancer', 
        'Chronic Kidney Disease', 'Depression', 'Epilepsy', 
        'Gastroesophageal Reflux Disease (GERD)', 'Hepatitis', 'HIV/AIDS', 
        'Hyperthyroidism', 'Hypothyroidism', 'Migraine', 'Osteoarthritis', 
        'Parkinson\'s Disease', 'Rheumatoid Arthritis', 'Tuberculosis', 
        'Ulcerative Colitis'
    ],
    'Health Precautions': [
        'Monitor blood sugar levels regularly', 
        'Reduce salt intake, regular exercise', 
        'Avoid smoking, manage stress', 
        'Avoid allergens, use inhalers as prescribed', 
        'Regular iron supplements, avoid tea/coffee with meals', 
        'Regular physical activity, portion control', 
        'Weight-bearing exercises, avoid excessive alcohol', 
        'Avoid allergens, use antihistamines', 
        'Maintain a healthy weight, regular exercise', 
        'Screening tests, avoid tobacco', 
        'Regular dialysis, fluid management', 
        'Medication adherence, regular therapy', 
        'Seizure control, medication adherence', 
        'Avoid trigger foods, antacids', 
        'Vaccination, safe sex practices', 
        'Medication adherence, regular check-ups', 
        'Medication adherence, stress management', 
        'Pain management, regular exercise', 
        'Medication adherence, physical therapy', 
        'Medication adherence, regular check-ups', 
        'Medication adherence, stress management', 
        'Medication adherence, regular exercise',
        'Avoid close contact: Stay away from people with active TB.',
        'Healthy diet: Maintain a balanced diet.'

    ],
    'Dietary Recommendations': [
        'Low-carb diet, high-fiber foods', 
        'Low-sodium diet, fruits, and vegetables', 
        'Low-fat diet, omega-3 fatty acids', 
        'Anti-inflammatory foods, avoid dairy', 
        'Iron-rich foods, vitamin C-rich foods', 
        'Balanced diet, low-calorie foods', 
        'Calcium-rich foods, vitamin D', 
        'Anti-inflammatory foods, avoid trigger foods', 
        'Omega-3 fatty acids, turmeric', 
        'High-fiber diet, antioxidants', 
        'Low-fat diet, high-fiber foods', 
        'Bland diet, avoid spicy foods', 
        'High-calorie diet, protein-rich foods', 
        'Low-sodium diet, potassium-rich foods', 
        'Iodine-rich foods, selenium-rich foods', 
        'Omega-3 fatty acids, vitamin D', 
        'Turmeric, ginger', 
        'Probiotic-rich foods, fiber-rich foods', 
        'Low-fat diet, antioxidants', 
        'High-fiber diet, vitamin D',
        'Antioxidants: Found in fruits, vegetables, and whole grains, these can help protect cells from damage.',
        'Vitamins and minerals: A balanced diet ensures adequate intake of essential vitamins and minerals.',
        'High-protein foods: To support tissue repair. Vitamins and minerals: Essential for overall health and immune function.',
        'Hydration: Ensure adequate fluid intake throughout the day. Fiber: While beneficial for overall health, it might need to be adjusted based on the specific condition.'
    ],
    'Exercise Recommendations': [
        'Brisk walking, swimming', 
        'Yoga, cycling', 
        'Cardio exercises, strength training', 
        'Breathing exercises, yoga', 
        'Walking, stretching', 
        'Swimming, water aerobics', 
        'Weight-bearing exercises, balance exercises', 
        'Yoga, Pilates', 
        'Low-impact aerobics, strength training', 
        'Breathing exercises, meditation', 
        'Walking, jogging', 
        'Yoga, tai chi', 
        'Swimming, cycling', 
        'High-intensity interval training, strength training', 
        'Pilates, yoga', 
        'Low-impact aerobics, flexibility exercises', 
        'Breathing exercises, relaxation techniques', 
        'Walking, stretching', 
        'Swimming, water aerobics', 
        'Weight-bearing exercises, balance exercises', 
        'Yoga, Pilates', 
        'Low-impact aerobics, stretching', 
        'Brisk walking, light jogging', 
        'Low-impact aerobics, strength training'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv('disease_recommendations8.csv', index=False)

# Preprocess data
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])
X = df[['Disease']]
y_health_precautions = df['Health Precautions']
y_dietary_recommendations = df['Dietary Recommendations']
y_exercise_recommendations = df['Exercise Recommendations']

# Split data for health precautions
X_train_hp, X_test_hp, y_train_hp, y_test_hp = train_test_split(X, y_health_precautions, test_size=0.2, random_state=42)

# Split data for dietary recommendations
X_train_dr, X_test_dr, y_train_dr, y_test_dr = train_test_split(X, y_dietary_recommendations, test_size=0.2, random_state=42)

# Split data for exercise recommendations
X_train_er, X_test_er, y_train_er, y_test_er = train_test_split(X, y_exercise_recommendations, test_size=0.2, random_state=42)

# Train model for Health Precautions
model_hp = RandomForestClassifier(n_estimators=100, random_state=42)
model_hp.fit(X_train_hp, y_train_hp)

# Train model for Dietary Recommendations
model_dr = RandomForestClassifier(n_estimators=100, random_state=42)
model_dr.fit(X_train_dr, y_train_dr)

# Train model for Exercise Recommendations
model_er = RandomForestClassifier(n_estimators=100, random_state=42)
model_er.fit(X_train_er, y_train_er)

# Save models
joblib.dump(model_hp, 'recommendation_model_hp8.pkl')
joblib.dump(model_dr, 'recommendation_model_dr8.pkl')
joblib.dump(model_er, 'recommendation_model_er8.pkl')
joblib.dump(le, 'label_encoder8.pkl')

print("Models trained and saved successfully.")
