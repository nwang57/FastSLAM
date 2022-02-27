import numpy as np

class EKF:
    def __init__(self, init_x, init_y, init_yaw):
        self.state = np.array([[init_x], [init_y], [init_yaw]])
        self.P = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self.noise = {'velocity': 0.2, 'yaw': 0.05, 'steer': 0.1, 'lidar': 0.5}

        self.dt = 1

    def update(self, state_vec):
        measured_x, measured_y, measured_yaw, vel, gamma = state_vec
    
        pred_state, pred_p = self.predict_step([vel, gamma])
        self.state, self.P = self.update_step([measured_x, measured_y, measured_yaw], [pred_state, pred_p])

        return self.state

    
    def predict_step(self, control):
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        B = self.getB()

        state = self.state
        p = self.P

        pred_state = A@state + B@control + self.get_control_noise()
        pred_p = A@p@np.transpose(A) + abs(self.get_control_noise())
        
        return pred_state, pred_p

    def update_step(self, measurement, pred):
        x_pose, y_pose, yaw = measurement
        pred_state, pred_p = pred

        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        z_k = np.array([[x_pose], [y_pose], [yaw]])

        y_k = z_k - H@pred_state
        S_k = H@pred_p@np.transpose(H) + abs(self.get_update_noise())
        print("S_k : ", S_k)
        print("det(S_k) : ", np.linalg.det(S_k))
        K = pred_p@np.transpose(H)@np.linalg.inv(S_k)
        print("K : ", K)
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

        vec = [[v_noise1], [v_noise2], [steer_noise]]

        return np.array(vec)

    def get_update_noise(self):
        x_pose_noise = np.random.normal(loc=0.0, scale=self.noise['lidar'])
        y_pose_noise = np.random.normal(loc=0.0, scale=self.noise['lidar'])
        yaw_noise = np.random.normal(loc=0.0, scale=self.noise['steer'])

        vec = [[x_pose_noise], [y_pose_noise], [yaw_noise]]

        return np.array(vec)