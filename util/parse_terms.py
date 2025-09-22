"""
File to store patterns for labeling heart patients.
"""

import re
from pathlib import Path
import pandas as pd


# 1. General cardiac words/prefixes
CARDIAC = [
    'cardiac', 'cardio', 'cardiology', 'cardiolgy', 
    'coronary', 'coronry',
    'myocardial', 'myocarial', 'heart attack', 'infarction', 'clot ', 
    'cardiovascular', 'cardiovasc', 'cvd', 'atrial', 'artery', 'aort', 'aneur',
]

# 2. Units/settings/roles/departments
UNITS = [
    'cardiac unit', 'heart unit', 'coronary unit', 'cardiology unit',
    ' CCU', 'coronary care unit', 'cardiac care unit',
    'CICU', 'cardiac intensive care unit',
    'cardiac rehab', 'cardiac rehabilitation', 'cardiac rehab unit',
    'cardiology department', 'cardiac department', 'heart department',
    'electrophysiology', ' EP lab', ' EP study', 'EP unit', ' EP '
]

# 3. Diseases/conditions/syndromes
DISEASES = [
    'heart disease', 'cardiac disease', 'coronary artery disease',
    'ischemic heart disease', 'IHD',
    'myocardial infarction',
    'arrhythmia', 'arrhythmic', 'atrial fibrillation', 
    'atrial flutter', 'ventricular tachycardia', 'flutter'
    'fib', 'fibrillation', 'atrial', 'A-fib', 'afib', 'heart af', 
    ' CHF ', 'congestive heart failure', ' hf ', 'heart failure',
    'a-flut', 'aflut', 'atrial flutter', 'fluttering',
    'supraventricular tachycardia', 'SVT', 
    'ventricular', 'tachycardia', 'VT', 'v-tach', 'vtach',
    'VF', 'ventricular fibrillation', 'v-fib', 'vfib',
    'complete heart block', 'heart block', ' chb ',
    'cardiomyopathy', 'CMP', 'dilated cardiomyopathy', 'myopathy',
    'hypertrophic cardiomyopathy', 'HCM', 'hypertrophic', 
    'restrictive cardiomyopathy', 'RCM', 'restrictive',
    'dilated cardiomyopathy', 'DCM',
    'peripartum cardiomyopathy', 'PPCM', 'peripartum',
    'arrhy', 'arrhyth', 'arrythmogenic', 'ARVC', 'arrhythmogenic right ventricular cardiomyopathy',
    'ventricular', 'cardiomyopathy', 'right ventricular', 'left ventricular',
    'rheumatic heart disease', ' RHD ', 'rheumatic',
    'aortic', 'aortic stenosis', 'aortic regurgitation', 'regurgitation',
    'mitral', 'mitral stenosis', 'mitral', 'regurg',
    'valve prolapse','pericarditis', 'pericardial', 'pericardium',
    'endocarditis', 'endocardial', 'endocardium',
    'thrombosis', 'thrombus', 'embolism', 'embolus',
    'aneurysm', 'aortic aneurysm', ' AAA ', 'abdominal aortic aneurysm',
    'TAA','arteriosclerosis', 'atherosclerosis', 'arterial sclerosis',
    'arterial', 'artery', 'arteries', 'arterial disease',
    'peripheral artery disease', 'peripheral vascular disease', ' PVD ',
    'vasculitis', 'vascular disease', 'vascular',
    'vasospasm', 'vasospastic', 'vasospasms',
    'vasoconstriction', 'vasodilator', 'vasodilation',
    'vasoconstrictor', 'vasodilators', 'vasodilating',
    'irregular heartbeat', 'palpitations', 'arrhythmias',
    'tachycardia', 'bradycardia', 'tachy', 'takotsubo', 'tamponade', 
    'tachyarrhythmia', 'bradyarrhythmia', 'tachycardias', 'bradycardias',
    'LBBB', 'left bundle branch block', 'RBBB', 'right bundle branch block',
    'LAFB', 'left anterior fascicular block', 'LPFB', 'left posterior fascicular block',
    'bifascicular', 'trifascicular', 'fascicular',
    'WPW', 'PSVT', 'AVB',
]

# 4. Procedures
PROCEDURES = [
    'angioplasty',
    'PCI', 'percutaneous coronary intervention', 'coronary', 'angioplast', 'sternotomy', 'stern'
    'bypass', 'by pass', 'CABG', 'cabbage', 'coronary artery bypass graft',
    'TAVR', 'TAVI', 'transcatheter',
    'cardiac ablation', 'ablation', 'EP ablation', 'electrophysiology',
    'cardioversion', 'defibrillation', 'defib', 'ICD', 'defibrillator',
    'pacemaker', 'cardioverter', 'PPM', 
    'ICD', 'CRT', 'pacer',
    'heart pump', 'assist device', 
    'heart transplant', 'heart transplant surgery', 'OHT', 
    'ventricular assist device', ' VAD ', 'left ventricular assist device', 'LVAD',
    'right ventricular assist device', 'RVAD', 'biventricular assist device',
    'BVAD', 'total artificial heart', ' TAH ',
    'heart valve replacement', 'valve replacement', 'valve repair',
    'valvuloplasty', 'valve surgery',
    'watchman', 'impella', 'micra', 'lifevest',
    'LAAO', 'LAAM', 'septal', 'myectomy', 'ablation', 'pericard', 
    'rotabl', 'POBA',
]

# 5. Diagnostics
DIAGNOSTICS = [
    'echocardiogram', 'cardiac ultrasound',
    'ECG', 'EKG', 'electrocardiogram', 
    'holter', ' TIA'
    'transtho',
    'stress test', 'exercise test', 'treadmill test', 'nuclear stress',
    'cardiac MRI', 'cMRI', 'cardiac CT', ' cCT', 'coronary CT',
]

# 6. Symptoms
SYMPTOMS = [
    'angina', 'anginal',
    'palpitations', 'fluttering',
    'orthopnea', 'PND', 'palpi', 'syncope'
]

# 7. Medications
MEDICATIONS = [
    'beta blocker', 'beta-blocker',
    'bisop', 
    'acei', 'lisino',
    'water pill', 'diuretic', 'diure', 'furo',
    'antiplate', 
    'blood thinner', 'antico', 'wafar', 'clop', 
]

# 8. Negative terms that are seemingly heart-related but are not
NEGATIVE_TERMS = [
    'heartfelt', 'heartwarming', 'my heart', 'our heart', 'broken heart', 'broke my heart', 'heart pillow'
    'bottom of our heart', 'bottom of my heart', 'heart health'
]

# print terms count
terms = set(CARDIAC + UNITS + DISEASES + PROCEDURES + DIAGNOSTICS + SYMPTOMS + MEDICATIONS + NEGATIVE_TERMS)
print(f"Total unique terms: {len(terms)}")




csv_path = Path("/home/minhle/projects/aip-btaati/minhle/Test/2023-08-01 - Pivot of comments by valence - MBC_Comments.csv")
date_cols = ["Experience Date", "Date of first reading"]

df = pd.read_csv(csv_path)
for col in date_cols:
    df[col] = pd.to_datetime(df[col].astype(str).str.strip(), format="mixed", errors="coerce")


# Check all comment for heart-related terms
def contains_heart_terms(comment: str):
    output = set()
    for term in CARDIAC + UNITS + DISEASES + PROCEDURES + DIAGNOSTICS + SYMPTOMS + MEDICATIONS:
        if term.lower() in comment.lower():
           output.add(term.strip())
    return list(output) if output else None


# Iterate through df and check comments

all_comments = df['Comment'].astype(str).tolist()

heart_df = {'Comment': [], 'Heart Terms': []}
non_heart_df = {'Comment': []}

for comment in all_comments:
    heart_terms = contains_heart_terms(comment)
    if heart_terms and not any(term.lower() in comment.lower() for term in NEGATIVE_TERMS):
        # remove newlines
        comment = comment.replace('\n', ' ').replace('\r', ' ').strip()
        heart_df['Comment'].append(comment)
        heart_df['Heart Terms'].append('|'.join(heart_terms))
    else:
        comment = comment.replace('\n', ' ').replace('\r', ' ').strip()
        non_heart_df['Comment'].append(comment)





# print 10 random heart-related comments and their heart terms
import numpy as np

print('Total', len(heart_df['Comment']), 'heart-related comments')

all_heart_terms = set()
for terms in heart_df['Heart Terms']:
    for term in terms.split('|'):
        all_heart_terms.add(term.strip())

print('Unique heart terms found:', len(all_heart_terms))

# save df to json
heart_path = Path("data/heart_related_comments.json")
pd.DataFrame(heart_df).to_json(heart_path, orient='records', lines=True)
print(f"Saved heart-related comments to {heart_path.absolute()}")


non_heart_path = Path("data/non_heart_related_comments.json")
pd.DataFrame(non_heart_df).to_json(non_heart_path, orient='records', lines=True)
print(f"Saved non-heart-related comments to {non_heart_path.absolute()}")

