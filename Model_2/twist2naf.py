import numpy as np
import matplotlib.pyplot as plt

class TwistModel:
    def __init__(self, mass_cylinder=2.5, radius_cylinder=0.2, length_cylinder=0.5, gravity=9.81, initial_height=10.0, damping=0.1, max_twist=np.pi/4, spring_constant=10.0):
        # Properties of each cylinder
        self.mass_cylinder = mass_cylinder  # Mass of each cylinder (kg)
        self.radius_cylinder = radius_cylinder  # Radius of each cylinder (m)
        self.length_cylinder = length_cylinder  # Length of each cylinder (m)
        self.gravity = gravity  # Gravitational acceleration (m/s^2)
        self.damping = damping  # Damping coefficient for rotational motion
        self.max_twist = max_twist  # Maximum allowable twist angle (radians)
        self.spring_constant = spring_constant  # Spring constant for restoring torque

        # Moments of inertia for twisting motion of each cylinder
        self.inertia = 0.5 * self.mass_cylinder * self.radius_cylinder**2

        # Initial conditions for twisting motion
        self.twist_angle_1 = 0.1  # Initial twisting angle of cylinder 1 (radians)
        self.twist_angle_2 = -0.1  # Initial twisting angle of cylinder 2 (radians)
        self.twist_velocity_1 = 0.2  # Initial angular velocity of cylinder 1 (rad/s)
        self.twist_velocity_2 = -0.2  # Initial angular velocity of cylinder 2 (rad/s)

        # Vertical motion
        self.height = initial_height  # Initial height of the system (m)
        self.vertical_velocity = 0.0  # Initial vertical velocity (m/s)

    def compute_torques(self):
        """
        Compute the torques acting on the two cylinders due to the coupling spring and damping.
        """
        torque_1 = -self.spring_constant * (self.twist_angle_1 - self.twist_angle_2) - self.damping * self.twist_velocity_1
        torque_2 = -self.spring_constant * (self.twist_angle_2 - self.twist_angle_1) - self.damping * self.twist_velocity_2
        return torque_1, torque_2

    def update_twist(self, delta_time):
        """
        Update the twisting motion of both cylinders considering torques and coupling.
        :param delta_time: Time increment for the simulation.
        """
        # Compute torques
        torque_1, torque_2 = self.compute_torques()

        # Compute angular accelerations
        twist_acceleration_1 = torque_1 / self.inertia
        twist_acceleration_2 = torque_2 / self.inertia

        # Update angular velocities
        self.twist_velocity_1 += twist_acceleration_1 * delta_time
        self.twist_velocity_2 += twist_acceleration_2 * delta_time

        # Conserve angular momentum (ensure their interaction balances out)
        total_angular_momentum = self.inertia * self.twist_velocity_1 + self.inertia * self.twist_velocity_2
        correction_factor = total_angular_momentum / (2 * self.inertia)
        self.twist_velocity_1 -= correction_factor
        self.twist_velocity_2 -= correction_factor

        # Update angles
        self.twist_angle_1 += self.twist_velocity_1 * delta_time
        self.twist_angle_2 += self.twist_velocity_2 * delta_time

        # Apply the maximum twist constraint
        if self.twist_angle_1 > self.max_twist:
            self.twist_angle_1 = self.max_twist
            self.twist_velocity_1 = 0
        elif self.twist_angle_1 < -self.max_twist:
            self.twist_angle_1 = -self.max_twist
            self.twist_velocity_1 = 0

        if self.twist_angle_2 > self.max_twist:
            self.twist_angle_2 = self.max_twist
            self.twist_velocity_2 = 0
        elif self.twist_angle_2 < -self.max_twist:
            self.twist_angle_2 = -self.max_twist
            self.twist_velocity_2 = 0

    def update_height(self, delta_time):
        """
        Update the vertical position of the system.
        :param delta_time: Time increment for the simulation.
        """
        self.vertical_velocity += self.gravity * delta_time
        self.height -= self.vertical_velocity * delta_time

    def compute_total_energy(self):
        """
        Compute the total energy in the system: rotational kinetic energy and spring potential energy.
        """
        kinetic_energy = 0.5 * self.inertia * (self.twist_velocity_1**2 + self.twist_velocity_2**2)
        potential_energy = 0.5 * self.spring_constant * (self.twist_angle_1 - self.twist_angle_2)**2
        return kinetic_energy + potential_energy

    def has_hit_ground(self):
        """Check if the system has hit the ground."""
        return self.height <= 0

# Simulation parameters
mass_cylinder = 2.5  # Mass of each cylinder (kg)
radius_cylinder = 0.2  # Radius of each cylinder (m)
length_cylinder = 0.5  # Length of each cylinder (m)
gravity = 9.81  # Gravitational acceleration (m/s^2)
initial_height = 10.0  # Initial height of the system (m)
damping = 0.1  # Damping coefficient
simulation_time = 10  # Maximum simulation time (s)
delta_time = 0.01  # Time step (s)
max_twist = np.pi / 4  # Maximum allowable twist angle (radians)
spring_constant = 10.0  # Spring constant for restoring torque

def run_simulation():
    model = TwistModel(mass_cylinder=mass_cylinder, radius_cylinder=radius_cylinder, length_cylinder=length_cylinder,
                       gravity=gravity, initial_height=initial_height, damping=damping, max_twist=max_twist, spring_constant=spring_constant)

    times = [0]
    heights = [model.height]
    twist_angles_1 = [model.twist_angle_1]
    twist_angles_2 = [model.twist_angle_2]
    energies = [model.compute_total_energy()]

    print("Starting simulation...")

    while times[-1] < simulation_time:
        model.update_twist(delta_time)
        model.update_height(delta_time)

        times.append(times[-1] + delta_time)
        heights.append(model.height)
        twist_angles_1.append(model.twist_angle_1)
        twist_angles_2.append(model.twist_angle_2)
        energies.append(model.compute_total_energy())

        if model.has_hit_ground():
            print(f"Cat hit the ground at time: {times[-1]:.2f} seconds")
            break

    # Convert to numpy arrays for plotting
    times = np.array(times)
    heights = np.array(heights)
    twist_angles_1 = np.array(twist_angles_1)
    twist_angles_2 = np.array(twist_angles_2)
    energies = np.array(energies)

    # Plot twisting motion
    plt.figure(figsize=(8, 6))
    plt.plot(times, twist_angles_1, label="Twist Angle Cylinder 1")
    plt.plot(times, twist_angles_2, label="Twist Angle Cylinder 2")
    plt.axhline(y=max_twist, color='r', linestyle='--', label="Max Twist")
    plt.axhline(y=-max_twist, color='r', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.title("Twist Model: Angular Motion of Two Cylinders")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot vertical motion
    plt.figure(figsize=(8, 6))
    plt.plot(times, heights, label="Height (m)")
    plt.axhline(y=0, color='r', linestyle='--', label="Ground Level")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Twist Model: Vertical Motion")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot energy conservation
    plt.figure(figsize=(8, 6))
    plt.plot(times, energies, label="Total Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Twist Model: Energy Conservation")
    plt.legend()
    plt.grid()
    plt.show()

run_simulation()

