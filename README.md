 **Medicine authentication system**

 
This project is a comprehensive AI-powered Medicine Authentication System designed to detect counterfeit pharmaceutical products using a combination of deep learning, OCR,
and metadata validation techniques. It integrates a custom-trained CNN model that analyzes uploaded medicine packaging images to classify them as real or fake based on visual
characteristics. The system is trained and validated using multiple datasets, including: 
(1) a labeled image dataset of real vs. fake medicine packaging used for CNN model training
(2) a structured medicine metadata dataset containing brand, manufacturer, and product details
(3) a manufacturer registry database listing officially approved manufacturers
(4) a batch database containing batch numbers, validation status, and expiry dates.
In addition to image-based verification, the system extracts batch numbers
using OCR and cross-checks them against the batch database to validate authenticity, expiry status, and manufacturing consistency. It further enhances verification by matching
brand details with metadata and confirming manufacturer legitimacy through the registry. The system intelligently identifies issues such as invalid or mismatched batch numbers, 
unapproved manufacturers, discontinued products, or expired batches. Built as an interactive Streamlit web application, it provides real-time predictions, transparent reasoning,
and detailed result breakdowns. The project includes training notebooks, dataset organization, metadata files, and automated model downloads via GitHub Releases, making it a
reliable end-to-end solution for detecting suspicious or counterfeit medicines and ensuring safer pharmaceutical distribution.
