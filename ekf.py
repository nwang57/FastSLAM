#tryoing to fix sk
import numpy as np

class EKF:
    def __init__(self, init_x, init_y, init_yaw):
        self.state = np.array([[init_x], [init_y], [init_yaw]]) # initial state
        self.P = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]) # variance array
        self.noise = {'velocity': 0.2, 'yaw': 0.0, 'steer': 0.05, 'lidar': 0.5}

        self.dt = 1 #timestamp

    def update(self, state_vec):
        measured_x, measured_y, measured_yaw, vel, gamma = state_vec
    
        pred_state, pred_p = self.predict_step([[vel], [gamma]])
        self.state, self.P = self.update_step([measured_x, measured_y, measured_yaw], [pred_state, pred_p])

        self.state[2][0] = self.process_yaw(self.state[2][0])

        return self.state

    
    def predict_step(self, control):
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #numbers of states
        B = self.getB()

        state = self.state
        p = self.P

        pred_state = A@state + B@control + self.get_control_noise()

        pred_p = A@p@np.transpose(A) + abs(self.get_control_noise()) #pred state covarince

        return pred_state, pred_p

    def update_step(self, measurement, pred):
        x_pose, y_pose, yaw = measurement
        pred_state, pred_p = pred

        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #measurement matrix 
        #R_k = np.array([1,0,0], [0,1,0], [0,0,1]) #sensor measure noise covarince matrix

        z_k = np.array([[x_pose], [y_pose], [yaw]])

        y_k = z_k - H@pred_state

        S_k = H@pred_p@np.transpose(H) + abs(self.get_update_noise())

        K = pred_p@np.transpose(H)@np.linalg.pinv(S_k)

        update_state = pred_state + K@y_k
        update_p = (np.identity(3) - K@H)@pred_p

        return update_state, update_p

    def getB(self):
        B = np.array([[np.cos(self.state[2][0])*self.dt, 0], [np.sin(self.state[2][0])*self.dt, 0], [0, self.dt]])
        return B

    def get_control_noise(self):
        v_noise1 = np.random.normal(loc=0.0, scale=self.noise['velocity'])
        v_noise2 = np.random.normal(loc=0.0, scale=self.noise['velocity'])
        steer_noise = np.random.normal(loc=0.0, scale=self.noise['steer'])

        vec = [[v_noise1], [v_noise2], [steer_noise]] #porcess noise

        return np.array(vec)

    def get_update_noise(self):
        x_pose_noise = np.random.normal(loc=0.0, scale=self.noise['lidar'])
        y_pose_noise = np.random.normal(loc=0.0, scale=self.noise['lidar'])
        yaw_noise = np.random.normal(loc=0.0, scale=self.noise['steer'])

        vec = [[x_pose_noise], [y_pose_noise], [yaw_noise]] #sesnor noise

        return np.array(vec)

    def process_yaw(self, yaw):
        pi = 3.14159
        if yaw > pi and yaw < pi*2:
            return yaw - 2*pi
        elif yaw > 2*pi:
            return yaw - 2*pi
        return yaw