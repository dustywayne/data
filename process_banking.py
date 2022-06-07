import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

def main():
	# read independent variables
	df = pd.read_csv('banking/independent_variables.csv')
	train_x,train_loan,test_x,test_loan = build_independents(df)
	build_and_test_model(train_x,train_loan,test_x,test_loan)

def build_independents(df):

	age = np.array(df['age'], dtype=int)
	job = np.array(df['job'], dtype=str)

	job_admin = np.array([a == 'admin.' for a in job], dtype=bool)
	job_bluecollar = np.array([a == 'blue-collar' for a in job], dtype=bool)
	job_entrepeneur = np.array([a == 'entrepreneur' for a in job], dtype=bool)
	job_housemaid = np.array([a == 'housemaid' for a in job], dtype=bool)
	job_management = np.array([a == 'management' for a in job], dtype=bool)
	job_retired = np.array([a == 'retired' for a in job], dtype=bool)
	job_selfemployed = np.array([a == 'self-employed' for a in job], dtype=bool)
	job_services = np.array([a == 'services' for a in job], dtype=bool)
	job_student = np.array([a == 'student' for a in job], dtype=bool)
	job_technician = np.array([a == 'technician' for a in job], dtype=bool)
	job_unemployed = np.array([a == 'unemployed' for a in job], dtype=bool)
	job_unknown = np.array([a == 'unknown' for a in job], dtype=bool)

	marital = np.array(df['marital'], dtype=str)
	marital_divorced = np.array([a == 'divorced' for a in marital], dtype=bool)
	marital_married = np.array([a == 'married' for a in marital], dtype=bool)
	marital_single = np.array([a == 'single' for a in marital], dtype=bool)
	marital_unknown = np.array([a == 'unknown' for a in marital], dtype=bool)

	education = np.array(df['education'], dtype=str)
	edu_four_yr = np.array([a == 'basic.4y' for a in education], dtype=bool)
	edu_six_yr = np.array([a == 'basic.6y' for a in education], dtype=bool)
	edu_nine_yr = np.array([a == 'basic.9y' for a in education], dtype=bool)
	edu_high_school = np.array([a == 'high.school' for a in education], dtype=bool)
	edu_illiterate = np.array([a == 'illiterate' for a in education], dtype=bool)
	edu_prof_crs = np.array([a == 'professional.course' for a in education], dtype=bool)
	edu_uni = np.array([a == 'university.degree' for a in education], dtype=bool)
	edu_unknown = np.array([a == 'unknown' for a in education], dtype=bool)

	# read dependent variable
	loan = np.array(pd.read_csv('banking/dependent_variable.csv')['loan'], dtype=str)
	loan = np.array([a == 'yes' for a in loan])

	# divide into training and testing data
	train_age = age[:len(age)//4]
	train_job_admin = job_admin[:len(job_admin)//4]
	train_job_bluecollar = job_bluecollar[:len(job_bluecollar)//4]
	train_job_entrepeneur = job_entrepeneur[:len(job_entrepeneur)//4]
	train_job_housemaid = job_housemaid[:len(job_housemaid)//4]
	train_job_management = job_management[:len(job_management)//4]
	train_job_retired = job_retired[:len(job_retired)//4]
	train_job_selfemployed = job_selfemployed[:len(job_selfemployed)//4]
	train_job_services = job_services[:len(job_services)//4]
	train_job_student = job_student[:len(job_student)//4]
	train_job_technician = job_technician[:len(job_technician)//4]
	train_job_unemployed = job_unemployed[:len(job_unemployed)//4]
	train_job_unknown = job_unknown[:len(job_unknown)//4]
	train_marital_divorced = marital_divorced[:len(marital_divorced)//4]
	train_marital_married = marital_married[:len(marital_married)//4]
	train_marital_single = marital_single[:len(marital_single)//4]
	train_marital_unknown = marital_unknown[:len(marital_unknown)//4]
	train_edu_four_yr = edu_four_yr[:len(edu_four_yr)//4]
	train_edu_six_yr = edu_six_yr[:len(edu_six_yr)//4]
	train_edu_nine_yr = edu_nine_yr[:len(edu_nine_yr)//4]
	train_edu_high_school = edu_high_school[:len(edu_high_school)//4]
	train_edu_illiterate = edu_illiterate[:len(edu_illiterate)//4]
	train_edu_prof_crs = edu_prof_crs[:len(edu_prof_crs)//4]
	train_edu_uni = edu_uni[:len(edu_uni)//4]
	train_edu_unknown = edu_unknown[:len(edu_unknown)//4]
	train_loan = loan[:len(loan)//4]

	test_age = age[len(age)//4:]
	test_job_admin = job_admin[len(job_admin)//4:]
	test_job_bluecollar = job_bluecollar[len(job_bluecollar)//4:]
	test_job_entrepeneur = job_entrepeneur[len(job_entrepeneur)//4:]
	test_job_housemaid = job_housemaid[len(job_housemaid)//4:]
	test_job_management = job_management[len(job_management)//4:]
	test_job_retired = job_retired[len(job_retired)//4:]
	test_job_selfemployed = job_selfemployed[len(job_selfemployed)//4:]
	test_job_services = job_services[len(job_services)//4:]
	test_job_student = job_student[len(job_student)//4:]
	test_job_technician = job_technician[len(job_technician)//4:]
	test_job_unemployed = job_unemployed[len(job_unemployed)//4:]
	test_job_unknown = job_unknown[len(job_unknown)//4:]
	test_marital_divorced = marital_divorced[len(marital_divorced)//4:]
	test_marital_married = marital_married[len(marital_married)//4:]
	test_marital_single = marital_single[len(marital_single)//4:]
	test_marital_unknown = marital_unknown[len(marital_unknown)//4:]
	test_edu_four_yr = edu_four_yr[len(edu_four_yr)//4:]
	test_edu_six_yr = edu_six_yr[len(edu_six_yr)//4:]
	test_edu_nine_yr = edu_nine_yr[len(edu_nine_yr)//4:]
	test_edu_high_school = edu_high_school[len(edu_high_school)//4:]
	test_edu_illiterate = edu_illiterate[len(edu_illiterate)//4:]
	test_edu_prof_crs = edu_prof_crs[len(edu_prof_crs)//4:]
	test_edu_uni = edu_uni[len(edu_uni)//4:]
	test_edu_unknown = edu_unknown[len(edu_unknown)//4:]
	test_loan = loan[len(loan)//4:]

	train_x = np.array([train_age,train_job_admin,train_job_bluecollar,train_job_entrepeneur,train_job_housemaid,train_job_management,train_job_retired,train_job_selfemployed,train_job_services,train_job_student,train_job_technician,train_job_unemployed,train_job_unknown,train_marital_divorced,train_marital_married,train_marital_single,train_marital_unknown,train_edu_four_yr,train_edu_six_yr,train_edu_nine_yr,train_edu_high_school,train_edu_illiterate,train_edu_prof_crs,train_edu_uni,train_edu_unknown]).transpose()
	test_x = np.array([test_age,test_job_admin,test_job_bluecollar,test_job_entrepeneur,test_job_housemaid,test_job_management,test_job_retired,test_job_selfemployed,test_job_services,test_job_student,test_job_technician,test_job_unemployed,test_job_unknown,test_marital_divorced,test_marital_married,test_marital_single,test_marital_unknown,test_edu_four_yr,test_edu_six_yr,test_edu_nine_yr,test_edu_high_school,test_edu_illiterate,test_edu_prof_crs,test_edu_uni,test_edu_unknown]).transpose()

	return train_x,train_loan,test_x,test_loan 

def build_and_test_model(train_x,train_loan,test_x,test_loan):

	features = ['age','job_admin','job_bluecollar','job_entrepeneur','job_housemaid','job_management','job_retired','job_selfemployed','job_services','job_student','job_technician','job_unemployed','job_unknown','marital_divorced','marital_married','marital_single','marital_unknown','edu_four_yr','edu_six_yr','edu_nine_yr','edu_high_school','edu_illiterate','edu_prof_crs','edu_uni','edu_unknown']

	model = LogisticRegression(solver='liblinear', random_state=0)
	model.fit(train_x,train_loan)
	# print the coefficients, i.e. the weight of each independent variable
	print('coefficients: ')
	for i in range(len(features)):
		print('{}{}'.format((features[i]+':').ljust(30), model.coef_[0][i]))

	print('intercept:'.ljust(30) + str(model.intercept_[0]))
	print('training score:'.ljust(30) + str(model.score(train_x,train_loan)))
	print('\nclassification report:\n' + str(classification_report(test_loan, model.predict(test_x))))
	print('testing score:'.ljust(30) + str(model.score(test_x,test_loan)))

if __name__ == '__main__':
	main()
