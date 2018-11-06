/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//default_random_engine gen;
	num_particles = 500;
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	//weights = vect(num_particles, 1.0/num_particles);
	for (int i=0; i< num_particles; i++){
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);
		//particles[i].weight = weights[i];
		Particle p;
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1.0;
		particles.push_back(p);
    }
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	//default_random_engine gen;
	double std_x, std_y, std_theta;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);
	for (int i=0; i< num_particles; i++){
		double noise_x, noise_y, noise_theta;
		noise_x = dist_x(gen);
		noise_y = dist_y(gen);
		noise_theta = dist_theta(gen);
		double x_next, y_next, theta_next;
		theta_next = particles[i].theta + yaw_rate * delta_t;
		if (fabs(yaw_rate) > 0.00001) {
			x_next = particles[i].x + velocity / yaw_rate * (sin(theta_next) - sin(particles[i].theta));
			y_next = particles[i].y + velocity / yaw_rate * (-cos(theta_next) + cos(particles[i].theta)); 
        }else{
			x_next = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			y_next = particles[i].y + velocity * delta_t * sin(particles[i].theta);
        }
		x_next += noise_x;
		y_next += noise_y;
		theta_next += noise_theta;
		particles[i].x = x_next;
		particles[i].y = y_next;
		particles[i].theta = theta_next;
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this predicted landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i = 0; i < observations.size(); i++){
		double min_dist = numeric_limits<double>::max();
		for(unsigned int j = 0; j < predicted.size(); j++){
			double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if(distance < min_dist){
				min_dist = distance;
				observations[i].id = predicted[j].id;
            }
        }
    } 
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (int i=0; i<num_particles; i++){
		//int p_id = particles[i].id;
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		vector<LandmarkObs> predicted;
        for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
			int lm_id = map_landmarks.landmark_list[j].id_i;
			double lm_x = map_landmarks.landmark_list[j].x_f;
			double lm_y = map_landmarks.landmark_list[j].y_f;
			if(dist(lm_x, lm_y, p_x, p_y) < sensor_range) {
				predicted.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
            }
        }

		vector<LandmarkObs> transformed_obs;
		for (unsigned int k=0; k< observations.size(); k++){
			double x_map = p_x + cos(p_theta) * observations[k].x - sin(p_theta) * observations[k].y;
			double y_map = p_y + sin(p_theta) * observations[k].x + cos(p_theta) * observations[k].y;
			transformed_obs.push_back(LandmarkObs{ k, x_map, y_map });
        }

		dataAssociation(predicted, transformed_obs);
		particles[i].weight = 1.0;
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		for (unsigned int j=0; j<transformed_obs.size(); j++){
			int association_id = transformed_obs[j].id;
			double mu_x, mu_y;
			for (unsigned int k=0; k<predicted.size(); k++) {
				if(predicted[k].id == association_id){
					//cout<<association_id<<"\n";
					mu_x = predicted[k].x;
					mu_y = predicted[k].y;
					//cout<<"landmark x"<<mu_x<<"\n";
					//cout<<"landmark y"<<mu_y<<"\n";
 					//cout<<"observed x"<<transformed_obs[j].x<<"\n";
 					//cout<<"observed y"<<transformed_obs[j].y<<"\n";
					associations.push_back(association_id);
					sense_x.push_back(transformed_obs[j].x);
					sense_x.push_back(transformed_obs[j].y);
                }
            }
			//calculate weight
			double gauss_norm= (1.0/(2.0 * M_PI * sig_x * sig_y));
			//cout << transformed_obs[j].x-mu_x << "\n";
			//cout << transformed_obs[j].y-mu_y << "\n";
			double exponent= pow(transformed_obs[j].x-mu_x, 2)/(2 * pow(sig_x,2)) + pow(transformed_obs[j].y-mu_y, 2)/(2 * pow(sig_y,2));
			//cout << exponent << "\n";
			particles[i].weight *= gauss_norm * exp(-exponent);
			//cout<<particles[i].weight<<"\n";
        }
		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	vector<Particle> new_particles;
	// get all of the current weights
	vector<double> weights_;
	//normalize weight
	double max_weight_init = 0;
	for (int i = 0; i < num_particles; i++) {
		if(particles[i].weight > max_weight_init){
			max_weight_init = particles[i].weight ;
        }
    }
	for (int i = 0; i < num_particles; i++) {
		//weights_.push_back(particles[i].weight);
		weights_.push_back(particles[i].weight / max_weight_init);
    }
	weights = weights_;
	// generate random starting index for resampling wheel
	uniform_int_distribution<int> uniintdist(0, num_particles-1);
	auto index = uniintdist(gen);
	// get max weight
	double max_weight = *max_element(weights.begin(), weights.end());
	//cout << max_weight << "\n";
	// uniform random distribution [0.0, max_weight)
	uniform_real_distribution<double> unirealdist(0.0, max_weight);
	double beta = 0.0;
	// spin the resample wheel!
	for (int i = 0; i < num_particles; i++) {
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
        }
		new_particles.push_back(particles[index]);
    }
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
