import numpy as np
from pypuf.simulation import ArbiterPUF
from pypuf.io import random_inputs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class PUFDevice:
    def __init__(self, n_stages, m_ghost_bits, hamming_threshold):
        self.n = n_stages
        self.m = m_ghost_bits
        self.hamming_threshold = hamming_threshold  # Configurable Hamming distance threshold
        self.ghost_bit_indices = self.generate_separated_ghost_bits()  # Pre-determined and separated ghost bit indices
        self.puf = ArbiterPUF(n=n_stages, seed=1)  # Initialize a PUF instance with fixed seed
        self.C_nvm = np.random.randint(0, 2, n_stages + m_ghost_bits)  # Initialize the device ID (C_nvm)
        self.R1 = np.zeros(n_stages + m_ghost_bits, dtype=int)  # Register R1
        self.R2 = np.zeros(n_stages // 2, dtype=int)  # Register R2
        self.R3 = np.zeros(n_stages, dtype=int)  # Register R3

    def generate_separated_ghost_bits(self):
        """Generate ghost bit indices with separation between them."""
        ghost_bit_indices = set()
        while len(ghost_bit_indices) < self.m:
            candidate = np.random.randint(0, self.n + self.m)
            if all(abs(candidate - idx) > 1 for idx in ghost_bit_indices):  # Ensure separation
                ghost_bit_indices.add(candidate)
        return sorted(ghost_bit_indices)

    def generate_challenge(self):
        """Generate a challenge using LFSR-like pseudo-random generation."""
        return random_inputs(n=self.n + self.m, N=1, seed=np.random.randint(0, 1000))[0]

    def evaluate_response(self):
        """Evaluate the PUF response for the current R1 content."""
        mapped_challenge = self.map_challenge(self.R1)
        response = self.puf.eval(mapped_challenge.reshape(1, -1))[0]
        return response

    def map_challenge(self, full_challenge):
        """Map the full challenge (n + m bits) to the n-bit challenge used for PUF stages."""
        mapped_challenge = [full_challenge[i] for i in range(self.n + self.m) if i not in self.ghost_bit_indices]
        return np.array(mapped_challenge[:self.n])

    def authenticate(self, R2_from_server):
        """Perform authentication using the received R2 and current C_nvm."""
        self.R1 = self.C_nvm.copy()  # Load C_nvm into R1
        self.R3 = np.zeros(self.n, dtype=int)  # Reset R3

        # Generate n responses
        for i in range(self.n):
            self.R3[i] = self.evaluate_response()
            self.R1 = np.roll(self.R1, -1)  # Simulate LFSR shift

        # Compare first n/2 bits of R3 with R2
        hamming_distance = np.sum(self.R3[:self.n // 2] != R2_from_server)
        if hamming_distance > self.hamming_threshold:  # Use configurable threshold
            return False, None  # Authentication failed

        # Send last n/2 bits of R3 and update C_nvm
        self.C_nvm = self.R1.copy()  # Update C_nvm with current R1
        return True, self.R3[self.n // 2:]

class Server:
    def __init__(self, n_stages, m_ghost_bits, ghost_bit_indices, hamming_threshold):
        self.n = n_stages
        self.m = m_ghost_bits
        self.ghost_bit_indices = ghost_bit_indices  # Same ghost bit indices as the device
        self.hamming_threshold = hamming_threshold  # Configurable Hamming distance threshold
        self.ID_device = np.random.randint(0, 2, n_stages + m_ghost_bits)  # Initialize device ID (ID_device)

    def generate_R2_and_responses(self):
        """Generate R2 and full responses based on the current ID_device."""
        R1 = self.ID_device.copy()  # Load ID_device into LFSR
        R3_soft = np.zeros(self.n, dtype=int)

        # Generate n responses using the device model
        for i in range(self.n):
            mapped_challenge = self.map_challenge(R1)
            R3_soft[i] = ArbiterPUF(n=self.n).eval(mapped_challenge.reshape(1, -1))[0]
            R1 = np.roll(R1, -1)  # Simulate LFSR shift

        R2 = R3_soft[:self.n // 2]  # First n/2 bits of R3_soft
        return R2, R3_soft

    def authenticate_device(self, R3_from_device):
        """Validate the device using the received R3 last half."""
        R2, R3_soft = self.generate_R2_and_responses()
        hamming_distance = np.sum(R3_soft[self.n // 2:] != R3_from_device)

        if hamming_distance > self.hamming_threshold:  # Use configurable threshold
            print("Server: Authentication failed due to high Hamming distance.")
            return False

        # Update ID_device with the last state of the LFSR
        self.ID_device = np.roll(self.ID_device, -1)
        print("Server: Authentication successful.")
        return True

    def map_challenge(self, full_challenge):
        """Map the full challenge (n + m bits) to the n-bit challenge used for PUF stages."""
        mapped_challenge = [full_challenge[i] for i in range(self.n + self.m) if i not in self.ghost_bit_indices]
        return np.array(mapped_challenge[:self.n])

# Replay attack demonstration
def replay_attack(server, device, replay_challenge):
    """Simulate a replay attack where an attacker replays a previously used challenge."""
    print("Replay Attack Simulation:")

    # Replay the challenge
    device.R1 = replay_challenge.copy()
    success, R3_last_half = device.authenticate(server.generate_R2_and_responses()[0])

    if not success:
        print("Replay attack failed. Protocol prevents reuse of challenges.")
    else:
        print("Replay attack succeeded. This should not happen in practice with proper challenge update.")

# Enrollment process
def enrollment_phase(server, device, num_crps):
    """Collect CRPs for enrollment and train the soft model."""
    enrollment_challenges = []
    enrollment_responses = []

    for _ in range(num_crps):
        full_challenge = device.generate_challenge()
        device.R1 = full_challenge.copy()  # Load challenge into R1
        response = device.evaluate_response()
        mapped_challenge = device.map_challenge(full_challenge)

        enrollment_challenges.append(mapped_challenge)
        enrollment_responses.append(response)

    enrollment_challenges = np.array(enrollment_challenges)
    enrollment_responses = np.array(enrollment_responses)

    # Train the soft model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(enrollment_challenges.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(enrollment_challenges, enrollment_responses, epochs=50, batch_size=64, verbose=1, validation_split=0.2)

    print("Enrollment phase completed. Soft model trained.")
    return model

# CRP-based attack implementation
def train_attack_model(challenges, responses, input_size):
    """Train a neural network to model the PUF behavior."""
    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_size,)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(challenges, responses, epochs=50, batch_size=64, verbose=1, validation_split=0.2)

    return model

# Simulation parameters
n_stages = 64
m_ghost_bits = 20
hamming_threshold = 8  # Configurable threshold
num_authentications = 10
num_crps_for_enrollment = 10000
num_crps_for_attack = 5000

# Initialize PUF device and server with separated ghost bit indices
device = PUFDevice(n_stages, m_ghost_bits, hamming_threshold)
server = Server(n_stages, m_ghost_bits, device.ghost_bit_indices, hamming_threshold)

# Enrollment phase
soft_model = enrollment_phase(server, device, num_crps_for_enrollment)

# Authentication phase with eavesdropping
eavesdropped_crps = []
for _ in range(num_authentications):
    R2, _ = server.generate_R2_and_responses()
    success, R3_last_half = device.authenticate(R2)

    if success:
        print("Device: Authentication successful. Sending R3 last half to server.")
        server.authenticate_device(R3_last_half)

        # Eavesdrop CRP
        eavesdropped_crps.append((device.R1.copy(), R2, R3_last_half))
    else:
        print("Device: Authentication failed.")

# Prepare data for the CRP-based attack
attack_challenges = []
attack_responses = []

for crp in eavesdropped_crps:
    full_challenge, R2, R3_last_half = crp
    for i in range(len(R2)):
        attack_challenges.append(full_challenge)
        attack_responses.append(R2[i])
    for i in range(len(R3_last_half)):
        attack_challenges.append(full_challenge)
        attack_responses.append(R3_last_half[i])

attack_challenges = np.array(attack_challenges)
attack_responses = np.array(attack_responses)

# Train the attack model
attack_model = train_attack_model(attack_challenges, attack_responses, n_stages + m_ghost_bits)

# Evaluate the attack model
loss, accuracy = attack_model.evaluate(attack_challenges, attack_responses, verbose=0)
print(f"Attack model accuracy: {accuracy:.2f}")

# Replay attack demonstration
replay_challenge = device.generate_challenge()
replay_attack(server, device, replay_challenge)
