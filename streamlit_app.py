import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
st.set_page_config(
    page_title="Handwritten Digit Recognizer (PyTorch)",
    layout="wide",
    initial_sidebar_state="expanded"
)
NUM_CLASSES = 10
IMG_SIZE = 28

class CNN(nn.Module):
    """
    Defines the Convolutional Neural Network architecture in PyTorch.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=NUM_CLASSES)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 3 * 3)
        
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    Loads and preprocesses the MNIST dataset using torchvision.
    """
    st.write("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    return train_loader, test_loader
def train_model(model, train_loader, test_loader, epochs=5):
    """
    Trains the PyTorch model.
    """
    st.write("Training the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
    
    st.success("Model training complete!")
    return model
def preprocess_user_image(image_file):
    """
    Preprocesses the user-uploaded image for model prediction in PyTorch format.
    """
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = Image.open(image_file)
        image_array = np.array(image.convert('L'))
        image_array = 255 - image_array
        inverted_image = Image.fromarray(image_array)
        tensor_image = transform(inverted_image)
        tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None
def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Handwritten Digit Recognition System (PyTorch)")
    st.markdown("""
        This application demonstrates a deep learning model for recognizing handwritten digits.
        The model is a Convolutional Neural Network (CNN) built with PyTorch, trained on the MNIST dataset.
        
        You can train the model from scratch and then upload an image of a handwritten digit to get a prediction!
    """)
    with st.sidebar:
        st.header("Project Controls")
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if st.button("Train Model"):
            train_loader, test_loader = load_and_preprocess_data()
            model = CNN()
            
            # Train the model.
            trained_model = train_model(model, train_loader, test_loader)
            
            # Save the trained model state.
            torch.save(trained_model.state_dict(), "digit_recognizer_model.pth")
            st.session_state.model_trained = True
            
            st.success("Model trained and saved. You can now upload an image for prediction.")
            st.balloons()
            
        st.markdown("---")
        st.info("You only need to train the model once. The trained weights are saved locally.")
    
    st.markdown("---")

    try:
        model = CNN()
        model.load_state_dict(torch.load("digit_recognizer_model.pth"))
        model.eval() # Set model to evaluation mode
        st.success("Model loaded successfully. Ready for prediction!")
        uploaded_file = st.file_uploader(
            "Upload an image of a handwritten digit (PNG or JPG):",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", width=200)
            processed_image = preprocess_user_image(uploaded_file)
            
            if processed_image is not None:
                with torch.no_grad():
                    outputs = model(processed_image)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_digit = torch.argmax(probabilities, dim=1).item()
                
                st.markdown("### Prediction:")
                st.write(f"The model predicts this is the digit: **{predicted_digit}**")
                st.markdown("---")
                st.markdown("#### Confidence Score:")
                bar_data = probabilities.squeeze().numpy()
                st.bar_chart(bar_data)
                st.write(f"Confidence: **{probabilities.max().item():.2%}**")

    except FileNotFoundError:
        st.info("Please train the model using the button in the sidebar before you can make a prediction.")
    except Exception as e:
        st.error(f"An error occurred while loading or using the model. Please retrain it. Error: {e}")
        st.session_state.model_trained = False
if __name__ == "__main__":
    main()