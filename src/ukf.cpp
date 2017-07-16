#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.9;//*****************Modified************************

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.4;//*************Modified********************

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  //************************************************* Initialization ************************************
  // Initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // Set state dimension
  n_x_ = 5;

  // Set augmented dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Time when the state is true, in us
  time_us_ = 0.0;

  // State covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Initial NIS value for Radar
  NIS_radar_ = 0.0;

  // Initial NIS value for Laser
  NIS_laser_ = 0.0;
}
//******************** Code for Initialization was inspired by Lesson 30. "UKF Update Assignment 2"**********

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
// TODO: Complete this function! Make sure you switch between lidar and radar measurements.

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
 
//******************************************* Process Measurement ***********************************************
	if (!is_initialized_) {

		// First measurement
		cout << "begin ProcessMeasurement" << endl;

		time_us_ = meas_package.timestamp_;

		// State vector
		x_ << 1, 1, 1, 1, 0.1;

		// State covariance matrix
		P_ << 0.15, 0, 0, 0, 0,
			0, 0.15, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1;

		// Check if sensor input is from laser or radar and update initial values accordingly
		if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
			float x = meas_package.raw_measurements_(0);
			float y = meas_package.raw_measurements_(1);

			if (x == 0 && y == 0) {
				x_(0) = 0.01;
				x_(1) = 0.01;
			}
			else {
				x_(0) = x;
				x_(1) = y;
			}

		}
		else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			float rho, phi;
			rho = meas_package.raw_measurements_(0);
			phi = meas_package.raw_measurements_(1);

			// Converting polar co-ordinates to cartesian
			x_(0) = rho * cos(phi);
			x_(1) = rho * sin(phi);
		}

		// Initially set to false, set to true in first call of ProcessMeasurement
		is_initialized_ = true;

		return;
	}

	// Calculate time since last sensor reading
	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0; // expressed in seconds
	
	cout << "Delta t: " << dt << endl;
	
	time_us_ = meas_package.timestamp_;

	// Prediction
	Prediction(dt);

	// Update the state and covariance matrices
	if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

		UpdateLidar(meas_package); // LIDAR update
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

		UpdateRadar(meas_package); // RADAR update

		cout << "end ProcessMeasurement" << endl;
	}
}
//Code for Process Measurement was inspired by these GitHub users' submissions: "grodowski", "squarerock", "gersakbogdan"

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

	cout << "begin Prediction" << endl;
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	//************************************** Generate Sigma Points *********************************************
	
	//set lambda (Sigma point spreading parameter) for non-augmented sigma points
	lambda_ = 3 - n_x_;

	//create sigma point matrix
	MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1); // (state dimension, 2 * state dimension + 1)

	//calculate square root of P
	MatrixXd A = P_.llt().matrixL(); // matrixL returns a view of the lower triangular matrix L

	//set first column of sigma point matrix
	Xsig.col(0) = x_;

	//set remaining sigma points
	for (int i = 0; i < n_x_; i++)
	{
		Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
		Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
	}
	//****************** Generating Sigma Points code was copied from GitHub user "jeremy-shannon "**************

	
	//**************************************** Augment Sigma points ********************************************

	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	//set lambda for augmented sigma points
	lambda_ = 3 - n_aug_; // n_aug is augmented state dimension

	//create augmented mean state
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}
	//*********** Code for Augmenting Sigma points was inspired by work of GitHub user "gersakbogdan" **********

	
	//************************************* Predict Sigma points ************************************************
	
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		//extract values for better readability
		double p_x = Xsig_aug(0, i);	  // Position X
		double p_y = Xsig_aug(1, i);	  // Position Y
		double v = Xsig_aug(2, i);		  // Velocity
		double yaw = Xsig_aug(3, i);	  // Yaw
		double yawd = Xsig_aug(4, i);	  // Yaw acceleration
		double nu_a = Xsig_aug(5, i);	  // Noise
		double nu_yawdd = Xsig_aug(6, i); // Noise

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
		}
		else {
			px_p = p_x + v * delta_t * cos(yaw);
			py_p = p_y + v * delta_t * sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd * delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
		py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
		v_p = v_p + nu_a * delta_t;

		yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
		yawd_p = yawd_p + nu_yawdd * delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;			// only 5 sigma points are required
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}
	//************** Code for Predicting Sigma points was inspired by work of GitHub user "LeoAnnArbor" *******

	
	//************************************** Predict Mean and Covariance ****************************************
	
	// set weights
	double weight_0 = lambda_ / (lambda_ + n_aug_);
	weights_(0) = weight_0;
	for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
		double weight = 0.5 / (n_aug_ + lambda_);
		weights_(i) = weight;
	}

	// Predict state mean
	x_.fill(0.0);
	// Predict state covariance matrix
	P_.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) //iterate over sigma points
	{  
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	for (int i = 0; i < 2 * n_aug_ + 1; i++) //iterate over sigma points
	{
		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		P_ += weights_(i) * x_diff * x_diff.transpose();
	}

	cout << "x" << endl << x_ << endl;
	cout << "P" << endl << P_ << endl;

	cout << "end Prediction" << endl;
}
	//****** Code for Predicting Mean and Covariance was inspired by work of GitHub user "grodowski" *************

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the lidar NIS.
  */

	//********************************************** Update Lidar ************************************************

	VectorXd z = meas_package.raw_measurements_;

	// Lidar can only measure distance (px,py)
	int n_z = 2;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  

		// extract values for better readibility
		double p_x = Xsig_pred_(0, i); // Position X
		double p_y = Xsig_pred_(1, i); // Position Y

		// measurement model
		Zsig(0, i) = p_x; // Matrix of position X
		Zsig(1, i) = p_y; // Matrix of position Y
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z); // n_z - measurement of px, py
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;		// The residual reflects the discrepancy between 
													// the predicted measurement
													// and the actual measurement
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_laspx_ * std_laspx_, 0, // Laser measurement noise standard deviation position1 in m
		0, std_laspy_ * std_laspy_;  // Laser measurement noise standard deviation position2 in m
	S = S + R;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0); // fill - Sets all coefficients in this expression to \a value.
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		//residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse(); // inverse - inverse Reference to the matrix in which to store the inverse

	//residual
	VectorXd z_diff = z - z_pred; // (extracted measurement - mean predicted measurement)

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff; // state vector
	P_ = P_ - K * S * K.transpose(); // state covariance matrix

	// Calculate NIS
	NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff; // NIS - Normalized Innovation Squared 
}
//*********** Code for Updating Lidar was inspired by work of GitHub user "jeremy-shannon" **********************

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the radar NIS.
  */

	//******************************************** Update Radar ***********************************************

	VectorXd z = meas_package.raw_measurements_;

	//  Measuring velocity (drho), distance in polar coordinates (rho, phi)
	int n_z = 3;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		// extract values for better readibility
		double p_x = Xsig_pred_(0, i); // Position X
		double p_y = Xsig_pred_(1, i); // Position Y
		double v = Xsig_pred_(2, i);   // Velocity
		double yaw = Xsig_pred_(3, i); // Yaw
		
		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);      
		Zsig(1, i) = atan2(p_y, p_x);                   
		Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
										
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI; // M_PI - 3.14159265358979323846...
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_ * std_radphi_, 0,
		0, 0, std_radrd_ * std_radrd_;

	S = S + R;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

										
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
	while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	// Calculate NIS
	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
//*************** Code for Updating Radar was inspired by work of GitHub user "jeremy-shannon" *****************