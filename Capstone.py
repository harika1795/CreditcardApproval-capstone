{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "TPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Project proposal to predict credit card approval\n",
        "\n",
        "A bank's credit card department is one of the top adopters of data science. A top focus for the bank has always been acquiring new credit card customers. Giving out credit cards without doing proper research or evaluating applicants' creditworthiness is quite risky. The credit card department has been using a data-driven system for credit assessment called Credit Scoring for many years, and the model is known as an application scorecard. A credit card application's cutoff value is determined using the application scorecard, which also aids in estimating the applicant's level of risk. This decision is made based on strategic priority at a given time.\n",
        "\n",
        "\n",
        "Customers must fill out a form, either physically or online, to apply for a credit card. The application data is used to evaluate the applicant's creditworthiness. The decision is made using the application data in addition to the Credit Bureau Score, such as the FICO Score in the US or the CIBIL Score in India, and other internal information on the applicants. Additionally, the banks are rapidly taking a lot of outside data into account to enhance the caliber of credit judgements."
      ],
      "metadata": {
        "id": "5MOJE0Q5tLhE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Attribute Descriptions**:-\n",
        "\n",
        "Gender: This column indicates the gender of the individual, either male or female. It is a Categorical (binary).\n",
        "\n",
        "Car_Owner: This column indicates whether the individual owns a car or not, either Yes or No.It is a Categorical (binary).\n",
        "\n",
        "Propert_Owner: This column indicates whether the individual owns a property or not, either Yes or No.It is a Categorical (binary).\n",
        "\n",
        "Children: This column indicates the number of children the individual has.It is a Numerical(integer).\n",
        "\n",
        "Annual_income: This column contains the annual income of the individual.It is a Numerical (continuous).\n",
        "\n",
        "Type_Income: This column indicates the type of income the individual earns, such as salary or self-employed income.It is a Numerical (nominal).\n",
        "\n",
        "Education: This column indicates the level of education of the individual. It is a Categorical (ordinal).\n",
        "\n",
        "Marital_status: This column indicates the marital status of the individual, such as Single, Married, Divorced, etc. It is a Categorical (nominal).\n",
        "\n",
        "Housing_type: This column indicates the type of housing the individual lives in, such as a house or apartment. It is a Categorical (nominal).\n",
        "\n",
        "Birthday_count: This column contains the age of the individual.It is a Numerical(integer).\n",
        "\n",
        "Employed_days: This column indicates the number of days the individual has been employed.It is a Numerical(integer).\n",
        "\n",
        "Mobile_phone, Work_Phone, Phone, and Email_id: These columns contain contact information for the individual, such as mobile phone number, work phone number, home phone number, and email id.Mobile_phone, Work_Phone, Phone are Categorical (binary) and Email_id is Categorical (nominal).\n",
        "\n",
        "Type_Occupation: This column indicates the type of occupation of the individual, such as healthcare or education. It is a Categorical (nominal).\n",
        "\n",
        "Family_Members: This column indicates the number of family members the individual has.It is a Numeric (integer).\n",
        "\n",
        "Label: This column contains the label for credit card approval, either approved or not approved. It is a Categorical (binary).\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "6f5pu89avxEI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "2) How is it going to impact the banking sector?\n",
        "\n",
        "A) predicting credit card approval using machine learning can have a transformative impact on the banking sector by improving the customer experience, reducing risk, and increasing profitability.\n",
        "\n",
        "Improved customer experience: By using predictive models, banks can offer faster and more accurate credit card approvals, which can improve the overall customer experience. This can lead to increased customer satisfaction and loyalty.\n",
        "\n",
        "Reduced risk of credit defaults: Machine learning algorithms can help banks to accurately predict the likelihood of a client defaulting on their credit card payments. By identifying clients with lower credit risk profiles, banks can offer credit cards with lower interest rates, which can reduce the risk of defaults and ultimately improve the bank's profitability.\n",
        "\n",
        "Improved profitability: By reducing the risk of credit defaults and improving the efficiency of the underwriting process, banks can increase their profitability. This can help banks to offer better rates and benefits to their clients, leading to increased market share and revenue.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "e_yRRDHltLjj"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "3) If any, what is the gap in the knowledge or how your proposed method can be helpful if required in future for any bank in India.\n",
        "\n",
        "A) If there is a gap in the knowledge or process used by a bank in India for credit card approval, the proposed method can be helpful for banks in India by improving the accuracy of credit card approval predictions, enhancing credit risk management strategies, and improving efficiency. However, to fully leverage the benefits of our proposed method, banks will need to ensure that they have access to reliable and comprehensive data sources.\n",
        "\n"
      ],
      "metadata": {
        "id": "YgJSx_jdtMvE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Importing Libraries"
      ],
      "metadata": {
        "id": "W0fPkCgEw0WN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Importing libraries\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ],
      "metadata": {
        "id": "jxCMwTkotOup"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Ignoring Warnings"
      ],
      "metadata": {
        "id": "ih3mKOsiw51J"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import warnings\n",
        "warnings.filterwarnings(\"ignore\")# ignoring warnings"
      ],
      "metadata": {
        "id": "IEqJ5JTOfAND"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Collection"
      ],
      "metadata": {
        "id": "j_xkCKGyw_cC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit=pd.read_csv(\"Credit_card.csv\")# reading url dataset\n",
        "credit_lab=pd.read_csv(\"Credit_card_label.csv\")# reading url dataset\n",
        "credit_card=pd.merge(credit,credit_lab,on='Ind_ID')#merging the dataset on common column"
      ],
      "metadata": {
        "id": "ZgsuokFFfDMi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw=credit_card.copy()# creating a copy of read file"
      ],
      "metadata": {
        "id": "DBf9eO1PfPUi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Preprocessing"
      ],
      "metadata": {
        "id": "WiDNTdlZxHsg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.shape # exploring number of observations and variables"
      ],
      "metadata": {
        "id": "vMGlVvpbfPsF",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f6fd8da7-3f22-4b21-f290-4174a442c3af"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1548, 19)"
            ]
          },
          "metadata": {},
          "execution_count": 155
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "We have 1548 observations and 19 variables."
      ],
      "metadata": {
        "id": "_ubuP6xFfWLa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.head()# head function to identify any erroneous value\n",
        "# Aim is to check if data looks fine"
      ],
      "metadata": {
        "id": "9ms2j_iNfRmX",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 313
        },
        "outputId": "3dab0426-6121-4803-cd01-b1284227b0a7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "    Ind_ID GENDER Car_Owner Propert_Owner  CHILDREN  Annual_income  \\\n",
              "0  5008827      M         Y             Y         0       180000.0   \n",
              "1  5009744      F         Y             N         0       315000.0   \n",
              "2  5009746      F         Y             N         0       315000.0   \n",
              "3  5009749      F         Y             N         0            NaN   \n",
              "4  5009752      F         Y             N         0       315000.0   \n",
              "\n",
              "            Type_Income         EDUCATION Marital_status       Housing_type  \\\n",
              "0             Pensioner  Higher education        Married  House / apartment   \n",
              "1  Commercial associate  Higher education        Married  House / apartment   \n",
              "2  Commercial associate  Higher education        Married  House / apartment   \n",
              "3  Commercial associate  Higher education        Married  House / apartment   \n",
              "4  Commercial associate  Higher education        Married  House / apartment   \n",
              "\n",
              "   Birthday_count  Employed_days  Mobile_phone  Work_Phone  Phone  EMAIL_ID  \\\n",
              "0        -18772.0         365243             1           0      0         0   \n",
              "1        -13557.0           -586             1           1      1         0   \n",
              "2             NaN           -586             1           1      1         0   \n",
              "3        -13557.0           -586             1           1      1         0   \n",
              "4        -13557.0           -586             1           1      1         0   \n",
              "\n",
              "  Type_Occupation  Family_Members  label  \n",
              "0             NaN               2      1  \n",
              "1             NaN               2      1  \n",
              "2             NaN               2      1  \n",
              "3             NaN               2      1  \n",
              "4             NaN               2      1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-94183166-ca72-44f1-a2a3-7ba1d9903ddf\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ind_ID</th>\n",
              "      <th>GENDER</th>\n",
              "      <th>Car_Owner</th>\n",
              "      <th>Propert_Owner</th>\n",
              "      <th>CHILDREN</th>\n",
              "      <th>Annual_income</th>\n",
              "      <th>Type_Income</th>\n",
              "      <th>EDUCATION</th>\n",
              "      <th>Marital_status</th>\n",
              "      <th>Housing_type</th>\n",
              "      <th>Birthday_count</th>\n",
              "      <th>Employed_days</th>\n",
              "      <th>Mobile_phone</th>\n",
              "      <th>Work_Phone</th>\n",
              "      <th>Phone</th>\n",
              "      <th>EMAIL_ID</th>\n",
              "      <th>Type_Occupation</th>\n",
              "      <th>Family_Members</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>5008827</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>180000.0</td>\n",
              "      <td>Pensioner</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-18772.0</td>\n",
              "      <td>365243</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>5009744</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13557.0</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>5009746</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>NaN</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5009749</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13557.0</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5009752</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13557.0</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-94183166-ca72-44f1-a2a3-7ba1d9903ddf')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-94183166-ca72-44f1-a2a3-7ba1d9903ddf button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-94183166-ca72-44f1-a2a3-7ba1d9903ddf');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-b8e36bfc-5777-457a-b268-910d24b0b9e4\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-b8e36bfc-5777-457a-b268-910d24b0b9e4')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-b8e36bfc-5777-457a-b268-910d24b0b9e4 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 156
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "There are missing values in multiple columns in form of NaN.\n",
        "Type_Occupation column has many missing values.\n",
        "\n",
        "Label:\n",
        "\n",
        " * 0 is application approved\n",
        " * 1 is application rejected"
      ],
      "metadata": {
        "id": "50YnCR-2feA_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.tail()# tail function to identify any erroneous value\n",
        "# Aim is to check if data looks fine"
      ],
      "metadata": {
        "id": "YeMTdq04faBW",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 348
        },
        "outputId": "dd2c8d2b-71a3-4acc-dafb-bcc8c4dce8a4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       Ind_ID GENDER Car_Owner Propert_Owner  CHILDREN  Annual_income  \\\n",
              "1543  5028645      F         N             Y         0            NaN   \n",
              "1544  5023655      F         N             N         0       225000.0   \n",
              "1545  5115992      M         Y             Y         2       180000.0   \n",
              "1546  5118219      M         Y             N         0       270000.0   \n",
              "1547  5053790      F         Y             Y         0       225000.0   \n",
              "\n",
              "               Type_Income                      EDUCATION  \\\n",
              "1543  Commercial associate               Higher education   \n",
              "1544  Commercial associate              Incomplete higher   \n",
              "1545               Working               Higher education   \n",
              "1546               Working  Secondary / secondary special   \n",
              "1547               Working               Higher education   \n",
              "\n",
              "            Marital_status       Housing_type  Birthday_count  Employed_days  \\\n",
              "1543               Married  House / apartment        -11957.0          -2182   \n",
              "1544  Single / not married  House / apartment        -10229.0          -1209   \n",
              "1545               Married  House / apartment        -13174.0          -2477   \n",
              "1546        Civil marriage  House / apartment        -15292.0           -645   \n",
              "1547               Married  House / apartment        -16601.0          -2859   \n",
              "\n",
              "      Mobile_phone  Work_Phone  Phone  EMAIL_ID Type_Occupation  \\\n",
              "1543             1           0      0         0        Managers   \n",
              "1544             1           0      0         0     Accountants   \n",
              "1545             1           0      0         0        Managers   \n",
              "1546             1           1      1         0         Drivers   \n",
              "1547             1           0      0         0             NaN   \n",
              "\n",
              "      Family_Members  label  \n",
              "1543               2      0  \n",
              "1544               1      0  \n",
              "1545               4      0  \n",
              "1546               2      0  \n",
              "1547               2      0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-325c8a59-9282-4d0f-9586-73f75d4ab2a2\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ind_ID</th>\n",
              "      <th>GENDER</th>\n",
              "      <th>Car_Owner</th>\n",
              "      <th>Propert_Owner</th>\n",
              "      <th>CHILDREN</th>\n",
              "      <th>Annual_income</th>\n",
              "      <th>Type_Income</th>\n",
              "      <th>EDUCATION</th>\n",
              "      <th>Marital_status</th>\n",
              "      <th>Housing_type</th>\n",
              "      <th>Birthday_count</th>\n",
              "      <th>Employed_days</th>\n",
              "      <th>Mobile_phone</th>\n",
              "      <th>Work_Phone</th>\n",
              "      <th>Phone</th>\n",
              "      <th>EMAIL_ID</th>\n",
              "      <th>Type_Occupation</th>\n",
              "      <th>Family_Members</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1543</th>\n",
              "      <td>5028645</td>\n",
              "      <td>F</td>\n",
              "      <td>N</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-11957.0</td>\n",
              "      <td>-2182</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Managers</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1544</th>\n",
              "      <td>5023655</td>\n",
              "      <td>F</td>\n",
              "      <td>N</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>225000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Incomplete higher</td>\n",
              "      <td>Single / not married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-10229.0</td>\n",
              "      <td>-1209</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Accountants</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1545</th>\n",
              "      <td>5115992</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>2</td>\n",
              "      <td>180000.0</td>\n",
              "      <td>Working</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13174.0</td>\n",
              "      <td>-2477</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Managers</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1546</th>\n",
              "      <td>5118219</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>270000.0</td>\n",
              "      <td>Working</td>\n",
              "      <td>Secondary / secondary special</td>\n",
              "      <td>Civil marriage</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-15292.0</td>\n",
              "      <td>-645</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>Drivers</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1547</th>\n",
              "      <td>5053790</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>225000.0</td>\n",
              "      <td>Working</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-16601.0</td>\n",
              "      <td>-2859</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-325c8a59-9282-4d0f-9586-73f75d4ab2a2')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-325c8a59-9282-4d0f-9586-73f75d4ab2a2 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-325c8a59-9282-4d0f-9586-73f75d4ab2a2');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-ad4abf24-8a32-4761-a173-1fc3967f4d8d\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-ad4abf24-8a32-4761-a173-1fc3967f4d8d')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-ad4abf24-8a32-4761-a173-1fc3967f4d8d button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 157
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "There are missing values in multiple columns in form of NaN.\n",
        "Index 1543 and 1547 has missing values."
      ],
      "metadata": {
        "id": "Rg9LnP4ofl5s"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.columns # checking column names\n",
        "# some column names to be renamed"
      ],
      "metadata": {
        "id": "7Nf-OQD7faE6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2092c245-b7c1-4f5d-bbbd-7c472495756d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['Ind_ID', 'GENDER', 'Car_Owner', 'Propert_Owner', 'CHILDREN',\n",
              "       'Annual_income', 'Type_Income', 'EDUCATION', 'Marital_status',\n",
              "       'Housing_type', 'Birthday_count', 'Employed_days', 'Mobile_phone',\n",
              "       'Work_Phone', 'Phone', 'EMAIL_ID', 'Type_Occupation', 'Family_Members',\n",
              "       'label'],\n",
              "      dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 158
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.dtypes # Checking if datatypes are read correctly.\n",
        "# We have to be sure about data types first and then match it with dtypes result.\n",
        "# If it is correct we are good otherewise we have to change the data type."
      ],
      "metadata": {
        "id": "0uqdeppUfaH5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3188b0da-181a-4efd-966b-c15241655417"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Ind_ID               int64\n",
              "GENDER              object\n",
              "Car_Owner           object\n",
              "Propert_Owner       object\n",
              "CHILDREN             int64\n",
              "Annual_income      float64\n",
              "Type_Income         object\n",
              "EDUCATION           object\n",
              "Marital_status      object\n",
              "Housing_type        object\n",
              "Birthday_count     float64\n",
              "Employed_days        int64\n",
              "Mobile_phone         int64\n",
              "Work_Phone           int64\n",
              "Phone                int64\n",
              "EMAIL_ID             int64\n",
              "Type_Occupation     object\n",
              "Family_Members       int64\n",
              "label                int64\n",
              "dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 159
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.nunique()# nuniuqe function to count unique values in each column\n"
      ],
      "metadata": {
        "id": "AGHdEN-tfaK0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6c1facd9-2ed7-46a0-f630-d72fab87eb64"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Ind_ID             1548\n",
              "GENDER                2\n",
              "Car_Owner             2\n",
              "Propert_Owner         2\n",
              "CHILDREN              6\n",
              "Annual_income       115\n",
              "Type_Income           4\n",
              "EDUCATION             5\n",
              "Marital_status        5\n",
              "Housing_type          6\n",
              "Birthday_count     1270\n",
              "Employed_days       956\n",
              "Mobile_phone          1\n",
              "Work_Phone            2\n",
              "Phone                 2\n",
              "EMAIL_ID              2\n",
              "Type_Occupation      18\n",
              "Family_Members        7\n",
              "label                 2\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 160
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "*   It is to check if each column has correct values, specifically categorical variable\n"
      ],
      "metadata": {
        "id": "sGjuY6Xgf4lN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.drop_duplicates() # check if there are duplicates"
      ],
      "metadata": {
        "id": "g7PTDSZ4faNt",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 652
        },
        "outputId": "1f08815f-c333-4a39-b609-cb3f5fe95f55"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       Ind_ID GENDER Car_Owner Propert_Owner  CHILDREN  Annual_income  \\\n",
              "0     5008827      M         Y             Y         0       180000.0   \n",
              "1     5009744      F         Y             N         0       315000.0   \n",
              "2     5009746      F         Y             N         0       315000.0   \n",
              "3     5009749      F         Y             N         0            NaN   \n",
              "4     5009752      F         Y             N         0       315000.0   \n",
              "...       ...    ...       ...           ...       ...            ...   \n",
              "1543  5028645      F         N             Y         0            NaN   \n",
              "1544  5023655      F         N             N         0       225000.0   \n",
              "1545  5115992      M         Y             Y         2       180000.0   \n",
              "1546  5118219      M         Y             N         0       270000.0   \n",
              "1547  5053790      F         Y             Y         0       225000.0   \n",
              "\n",
              "               Type_Income                      EDUCATION  \\\n",
              "0                Pensioner               Higher education   \n",
              "1     Commercial associate               Higher education   \n",
              "2     Commercial associate               Higher education   \n",
              "3     Commercial associate               Higher education   \n",
              "4     Commercial associate               Higher education   \n",
              "...                    ...                            ...   \n",
              "1543  Commercial associate               Higher education   \n",
              "1544  Commercial associate              Incomplete higher   \n",
              "1545               Working               Higher education   \n",
              "1546               Working  Secondary / secondary special   \n",
              "1547               Working               Higher education   \n",
              "\n",
              "            Marital_status       Housing_type  Birthday_count  Employed_days  \\\n",
              "0                  Married  House / apartment        -18772.0         365243   \n",
              "1                  Married  House / apartment        -13557.0           -586   \n",
              "2                  Married  House / apartment             NaN           -586   \n",
              "3                  Married  House / apartment        -13557.0           -586   \n",
              "4                  Married  House / apartment        -13557.0           -586   \n",
              "...                    ...                ...             ...            ...   \n",
              "1543               Married  House / apartment        -11957.0          -2182   \n",
              "1544  Single / not married  House / apartment        -10229.0          -1209   \n",
              "1545               Married  House / apartment        -13174.0          -2477   \n",
              "1546        Civil marriage  House / apartment        -15292.0           -645   \n",
              "1547               Married  House / apartment        -16601.0          -2859   \n",
              "\n",
              "      Mobile_phone  Work_Phone  Phone  EMAIL_ID Type_Occupation  \\\n",
              "0                1           0      0         0             NaN   \n",
              "1                1           1      1         0             NaN   \n",
              "2                1           1      1         0             NaN   \n",
              "3                1           1      1         0             NaN   \n",
              "4                1           1      1         0             NaN   \n",
              "...            ...         ...    ...       ...             ...   \n",
              "1543             1           0      0         0        Managers   \n",
              "1544             1           0      0         0     Accountants   \n",
              "1545             1           0      0         0        Managers   \n",
              "1546             1           1      1         0         Drivers   \n",
              "1547             1           0      0         0             NaN   \n",
              "\n",
              "      Family_Members  label  \n",
              "0                  2      1  \n",
              "1                  2      1  \n",
              "2                  2      1  \n",
              "3                  2      1  \n",
              "4                  2      1  \n",
              "...              ...    ...  \n",
              "1543               2      0  \n",
              "1544               1      0  \n",
              "1545               4      0  \n",
              "1546               2      0  \n",
              "1547               2      0  \n",
              "\n",
              "[1548 rows x 19 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a9f7452e-d0c5-4396-a4f8-1f178031cb29\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ind_ID</th>\n",
              "      <th>GENDER</th>\n",
              "      <th>Car_Owner</th>\n",
              "      <th>Propert_Owner</th>\n",
              "      <th>CHILDREN</th>\n",
              "      <th>Annual_income</th>\n",
              "      <th>Type_Income</th>\n",
              "      <th>EDUCATION</th>\n",
              "      <th>Marital_status</th>\n",
              "      <th>Housing_type</th>\n",
              "      <th>Birthday_count</th>\n",
              "      <th>Employed_days</th>\n",
              "      <th>Mobile_phone</th>\n",
              "      <th>Work_Phone</th>\n",
              "      <th>Phone</th>\n",
              "      <th>EMAIL_ID</th>\n",
              "      <th>Type_Occupation</th>\n",
              "      <th>Family_Members</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>5008827</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>180000.0</td>\n",
              "      <td>Pensioner</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-18772.0</td>\n",
              "      <td>365243</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>5009744</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13557.0</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>5009746</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>NaN</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5009749</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13557.0</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5009752</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13557.0</td>\n",
              "      <td>-586</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1543</th>\n",
              "      <td>5028645</td>\n",
              "      <td>F</td>\n",
              "      <td>N</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-11957.0</td>\n",
              "      <td>-2182</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Managers</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1544</th>\n",
              "      <td>5023655</td>\n",
              "      <td>F</td>\n",
              "      <td>N</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>225000.0</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Incomplete higher</td>\n",
              "      <td>Single / not married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-10229.0</td>\n",
              "      <td>-1209</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Accountants</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1545</th>\n",
              "      <td>5115992</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>2</td>\n",
              "      <td>180000.0</td>\n",
              "      <td>Working</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-13174.0</td>\n",
              "      <td>-2477</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Managers</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1546</th>\n",
              "      <td>5118219</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>270000.0</td>\n",
              "      <td>Working</td>\n",
              "      <td>Secondary / secondary special</td>\n",
              "      <td>Civil marriage</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-15292.0</td>\n",
              "      <td>-645</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>Drivers</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1547</th>\n",
              "      <td>5053790</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>225000.0</td>\n",
              "      <td>Working</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>-16601.0</td>\n",
              "      <td>-2859</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1548 rows × 19 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a9f7452e-d0c5-4396-a4f8-1f178031cb29')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-a9f7452e-d0c5-4396-a4f8-1f178031cb29 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-a9f7452e-d0c5-4396-a4f8-1f178031cb29');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-1dc57832-9ac4-459a-a87e-3b3a6f82f55c\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-1dc57832-9ac4-459a-a87e-3b3a6f82f55c')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-1dc57832-9ac4-459a-a87e-3b3a6f82f55c button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 161
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw[credit_card_raw.duplicated]"
      ],
      "metadata": {
        "id": "RprNDymYfaQw",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 73
        },
        "outputId": "19d526c1-51dc-41be-b0ec-e7ba9b7d7018"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Empty DataFrame\n",
              "Columns: [Ind_ID, GENDER, Car_Owner, Propert_Owner, CHILDREN, Annual_income, Type_Income, EDUCATION, Marital_status, Housing_type, Birthday_count, Employed_days, Mobile_phone, Work_Phone, Phone, EMAIL_ID, Type_Occupation, Family_Members, label]\n",
              "Index: []"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-aad69873-5970-48a2-b1e6-6bca7950c776\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ind_ID</th>\n",
              "      <th>GENDER</th>\n",
              "      <th>Car_Owner</th>\n",
              "      <th>Propert_Owner</th>\n",
              "      <th>CHILDREN</th>\n",
              "      <th>Annual_income</th>\n",
              "      <th>Type_Income</th>\n",
              "      <th>EDUCATION</th>\n",
              "      <th>Marital_status</th>\n",
              "      <th>Housing_type</th>\n",
              "      <th>Birthday_count</th>\n",
              "      <th>Employed_days</th>\n",
              "      <th>Mobile_phone</th>\n",
              "      <th>Work_Phone</th>\n",
              "      <th>Phone</th>\n",
              "      <th>EMAIL_ID</th>\n",
              "      <th>Type_Occupation</th>\n",
              "      <th>Family_Members</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-aad69873-5970-48a2-b1e6-6bca7950c776')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-aad69873-5970-48a2-b1e6-6bca7950c776 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-aad69873-5970-48a2-b1e6-6bca7950c776');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 162
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.describe()"
      ],
      "metadata": {
        "id": "L22OpqGPfaUH",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 320
        },
        "outputId": "f6a45a65-8c89-4789-da97-139a5e7418ce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "             Ind_ID     CHILDREN  Annual_income  Birthday_count  \\\n",
              "count  1.548000e+03  1548.000000   1.525000e+03     1526.000000   \n",
              "mean   5.078920e+06     0.412791   1.913993e+05   -16040.342071   \n",
              "std    4.171759e+04     0.776691   1.132530e+05     4229.503202   \n",
              "min    5.008827e+06     0.000000   3.375000e+04   -24946.000000   \n",
              "25%    5.045070e+06     0.000000   1.215000e+05   -19553.000000   \n",
              "50%    5.078842e+06     0.000000   1.665000e+05   -15661.500000   \n",
              "75%    5.115673e+06     1.000000   2.250000e+05   -12417.000000   \n",
              "max    5.150412e+06    14.000000   1.575000e+06    -7705.000000   \n",
              "\n",
              "       Employed_days  Mobile_phone   Work_Phone        Phone     EMAIL_ID  \\\n",
              "count    1548.000000        1548.0  1548.000000  1548.000000  1548.000000   \n",
              "mean    59364.689922           1.0     0.208010     0.309432     0.092377   \n",
              "std    137808.062701           0.0     0.406015     0.462409     0.289651   \n",
              "min    -14887.000000           1.0     0.000000     0.000000     0.000000   \n",
              "25%     -3174.500000           1.0     0.000000     0.000000     0.000000   \n",
              "50%     -1565.000000           1.0     0.000000     0.000000     0.000000   \n",
              "75%      -431.750000           1.0     0.000000     1.000000     0.000000   \n",
              "max    365243.000000           1.0     1.000000     1.000000     1.000000   \n",
              "\n",
              "       Family_Members        label  \n",
              "count     1548.000000  1548.000000  \n",
              "mean         2.161499     0.113049  \n",
              "std          0.947772     0.316755  \n",
              "min          1.000000     0.000000  \n",
              "25%          2.000000     0.000000  \n",
              "50%          2.000000     0.000000  \n",
              "75%          3.000000     0.000000  \n",
              "max         15.000000     1.000000  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fcef3dcf-0b53-4b36-bf15-34ca6aee6088\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ind_ID</th>\n",
              "      <th>CHILDREN</th>\n",
              "      <th>Annual_income</th>\n",
              "      <th>Birthday_count</th>\n",
              "      <th>Employed_days</th>\n",
              "      <th>Mobile_phone</th>\n",
              "      <th>Work_Phone</th>\n",
              "      <th>Phone</th>\n",
              "      <th>EMAIL_ID</th>\n",
              "      <th>Family_Members</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>1.548000e+03</td>\n",
              "      <td>1548.000000</td>\n",
              "      <td>1.525000e+03</td>\n",
              "      <td>1526.000000</td>\n",
              "      <td>1548.000000</td>\n",
              "      <td>1548.0</td>\n",
              "      <td>1548.000000</td>\n",
              "      <td>1548.000000</td>\n",
              "      <td>1548.000000</td>\n",
              "      <td>1548.000000</td>\n",
              "      <td>1548.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>5.078920e+06</td>\n",
              "      <td>0.412791</td>\n",
              "      <td>1.913993e+05</td>\n",
              "      <td>-16040.342071</td>\n",
              "      <td>59364.689922</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.208010</td>\n",
              "      <td>0.309432</td>\n",
              "      <td>0.092377</td>\n",
              "      <td>2.161499</td>\n",
              "      <td>0.113049</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>4.171759e+04</td>\n",
              "      <td>0.776691</td>\n",
              "      <td>1.132530e+05</td>\n",
              "      <td>4229.503202</td>\n",
              "      <td>137808.062701</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.406015</td>\n",
              "      <td>0.462409</td>\n",
              "      <td>0.289651</td>\n",
              "      <td>0.947772</td>\n",
              "      <td>0.316755</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>5.008827e+06</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.375000e+04</td>\n",
              "      <td>-24946.000000</td>\n",
              "      <td>-14887.000000</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>5.045070e+06</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.215000e+05</td>\n",
              "      <td>-19553.000000</td>\n",
              "      <td>-3174.500000</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>5.078842e+06</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.665000e+05</td>\n",
              "      <td>-15661.500000</td>\n",
              "      <td>-1565.000000</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>5.115673e+06</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>2.250000e+05</td>\n",
              "      <td>-12417.000000</td>\n",
              "      <td>-431.750000</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>5.150412e+06</td>\n",
              "      <td>14.000000</td>\n",
              "      <td>1.575000e+06</td>\n",
              "      <td>-7705.000000</td>\n",
              "      <td>365243.000000</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>15.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fcef3dcf-0b53-4b36-bf15-34ca6aee6088')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-fcef3dcf-0b53-4b36-bf15-34ca6aee6088 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-fcef3dcf-0b53-4b36-bf15-34ca6aee6088');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-6987bb26-0fd3-41c4-b9f2-8577dc9afc4a\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-6987bb26-0fd3-41c4-b9f2-8577dc9afc4a')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-6987bb26-0fd3-41c4-b9f2-8577dc9afc4a button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 163
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.columns"
      ],
      "metadata": {
        "id": "QCabpej0geHP",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "020f97f4-c160-4a22-e5b7-d992e634a869"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['Ind_ID', 'GENDER', 'Car_Owner', 'Propert_Owner', 'CHILDREN',\n",
              "       'Annual_income', 'Type_Income', 'EDUCATION', 'Marital_status',\n",
              "       'Housing_type', 'Birthday_count', 'Employed_days', 'Mobile_phone',\n",
              "       'Work_Phone', 'Phone', 'EMAIL_ID', 'Type_Occupation', 'Family_Members',\n",
              "       'label'],\n",
              "      dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 164
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Cleaning"
      ],
      "metadata": {
        "id": "qVSHpCE9MB_R"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.isnull().sum()"
      ],
      "metadata": {
        "id": "YEizL1ayld57",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fb542644-ccb1-4202-8ae2-ca2a37d4c4fe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Ind_ID               0\n",
              "GENDER               7\n",
              "Car_Owner            0\n",
              "Propert_Owner        0\n",
              "CHILDREN             0\n",
              "Annual_income       23\n",
              "Type_Income          0\n",
              "EDUCATION            0\n",
              "Marital_status       0\n",
              "Housing_type         0\n",
              "Birthday_count      22\n",
              "Employed_days        0\n",
              "Mobile_phone         0\n",
              "Work_Phone           0\n",
              "Phone                0\n",
              "EMAIL_ID             0\n",
              "Type_Occupation    488\n",
              "Family_Members       0\n",
              "label                0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 165
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw['age']=np.abs(credit_card_raw['Birthday_count']/365)\n",
        "credit_card_raw['experience']=np.abs(credit_card_raw['Employed_days']/365)"
      ],
      "metadata": {
        "id": "LJS_T2Anl_IO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw['Annual_income'].fillna(credit_card_raw['Annual_income'].mean(), inplace=True)\n",
        "credit_card_raw['age'].fillna(credit_card_raw['age'].mean(), inplace=True)"
      ],
      "metadata": {
        "id": "-h_loay9leHu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw=credit_card_raw.drop(['Birthday_count','Employed_days'],axis=1)"
      ],
      "metadata": {
        "id": "lM51KEcrmMdv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw"
      ],
      "metadata": {
        "id": "pe5MpZKCmdPH",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 652
        },
        "outputId": "acc36e2d-52c2-4a3c-8a6a-f8ee8f6431fd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       Ind_ID GENDER Car_Owner Propert_Owner  CHILDREN  Annual_income  \\\n",
              "0     5008827      M         Y             Y         0   180000.00000   \n",
              "1     5009744      F         Y             N         0   315000.00000   \n",
              "2     5009746      F         Y             N         0   315000.00000   \n",
              "3     5009749      F         Y             N         0   191399.32623   \n",
              "4     5009752      F         Y             N         0   315000.00000   \n",
              "...       ...    ...       ...           ...       ...            ...   \n",
              "1543  5028645      F         N             Y         0   191399.32623   \n",
              "1544  5023655      F         N             N         0   225000.00000   \n",
              "1545  5115992      M         Y             Y         2   180000.00000   \n",
              "1546  5118219      M         Y             N         0   270000.00000   \n",
              "1547  5053790      F         Y             Y         0   225000.00000   \n",
              "\n",
              "               Type_Income                      EDUCATION  \\\n",
              "0                Pensioner               Higher education   \n",
              "1     Commercial associate               Higher education   \n",
              "2     Commercial associate               Higher education   \n",
              "3     Commercial associate               Higher education   \n",
              "4     Commercial associate               Higher education   \n",
              "...                    ...                            ...   \n",
              "1543  Commercial associate               Higher education   \n",
              "1544  Commercial associate              Incomplete higher   \n",
              "1545               Working               Higher education   \n",
              "1546               Working  Secondary / secondary special   \n",
              "1547               Working               Higher education   \n",
              "\n",
              "            Marital_status       Housing_type  Mobile_phone  Work_Phone  \\\n",
              "0                  Married  House / apartment             1           0   \n",
              "1                  Married  House / apartment             1           1   \n",
              "2                  Married  House / apartment             1           1   \n",
              "3                  Married  House / apartment             1           1   \n",
              "4                  Married  House / apartment             1           1   \n",
              "...                    ...                ...           ...         ...   \n",
              "1543               Married  House / apartment             1           0   \n",
              "1544  Single / not married  House / apartment             1           0   \n",
              "1545               Married  House / apartment             1           0   \n",
              "1546        Civil marriage  House / apartment             1           1   \n",
              "1547               Married  House / apartment             1           0   \n",
              "\n",
              "      Phone  EMAIL_ID Type_Occupation  Family_Members  label        age  \\\n",
              "0         0         0             NaN               2      1  51.430137   \n",
              "1         1         0             NaN               2      1  37.142466   \n",
              "2         1         0             NaN               2      1  43.946143   \n",
              "3         1         0             NaN               2      1  37.142466   \n",
              "4         1         0             NaN               2      1  37.142466   \n",
              "...     ...       ...             ...             ...    ...        ...   \n",
              "1543      0         0        Managers               2      0  32.758904   \n",
              "1544      0         0     Accountants               1      0  28.024658   \n",
              "1545      0         0        Managers               4      0  36.093151   \n",
              "1546      1         0         Drivers               2      0  41.895890   \n",
              "1547      0         0             NaN               2      0  45.482192   \n",
              "\n",
              "       experience  \n",
              "0     1000.665753  \n",
              "1        1.605479  \n",
              "2        1.605479  \n",
              "3        1.605479  \n",
              "4        1.605479  \n",
              "...           ...  \n",
              "1543     5.978082  \n",
              "1544     3.312329  \n",
              "1545     6.786301  \n",
              "1546     1.767123  \n",
              "1547     7.832877  \n",
              "\n",
              "[1548 rows x 19 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-7154fd0e-ed50-416b-b831-cb92be611aac\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ind_ID</th>\n",
              "      <th>GENDER</th>\n",
              "      <th>Car_Owner</th>\n",
              "      <th>Propert_Owner</th>\n",
              "      <th>CHILDREN</th>\n",
              "      <th>Annual_income</th>\n",
              "      <th>Type_Income</th>\n",
              "      <th>EDUCATION</th>\n",
              "      <th>Marital_status</th>\n",
              "      <th>Housing_type</th>\n",
              "      <th>Mobile_phone</th>\n",
              "      <th>Work_Phone</th>\n",
              "      <th>Phone</th>\n",
              "      <th>EMAIL_ID</th>\n",
              "      <th>Type_Occupation</th>\n",
              "      <th>Family_Members</th>\n",
              "      <th>label</th>\n",
              "      <th>age</th>\n",
              "      <th>experience</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>5008827</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>180000.00000</td>\n",
              "      <td>Pensioner</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>51.430137</td>\n",
              "      <td>1000.665753</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>5009744</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.00000</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>37.142466</td>\n",
              "      <td>1.605479</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>5009746</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.00000</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>43.946143</td>\n",
              "      <td>1.605479</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5009749</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>191399.32623</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>37.142466</td>\n",
              "      <td>1.605479</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5009752</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.00000</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>37.142466</td>\n",
              "      <td>1.605479</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1543</th>\n",
              "      <td>5028645</td>\n",
              "      <td>F</td>\n",
              "      <td>N</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>191399.32623</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Managers</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>32.758904</td>\n",
              "      <td>5.978082</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1544</th>\n",
              "      <td>5023655</td>\n",
              "      <td>F</td>\n",
              "      <td>N</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>225000.00000</td>\n",
              "      <td>Commercial associate</td>\n",
              "      <td>Incomplete higher</td>\n",
              "      <td>Single / not married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Accountants</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>28.024658</td>\n",
              "      <td>3.312329</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1545</th>\n",
              "      <td>5115992</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>2</td>\n",
              "      <td>180000.00000</td>\n",
              "      <td>Working</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>Managers</td>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>36.093151</td>\n",
              "      <td>6.786301</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1546</th>\n",
              "      <td>5118219</td>\n",
              "      <td>M</td>\n",
              "      <td>Y</td>\n",
              "      <td>N</td>\n",
              "      <td>0</td>\n",
              "      <td>270000.00000</td>\n",
              "      <td>Working</td>\n",
              "      <td>Secondary / secondary special</td>\n",
              "      <td>Civil marriage</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>Drivers</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>41.895890</td>\n",
              "      <td>1.767123</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1547</th>\n",
              "      <td>5053790</td>\n",
              "      <td>F</td>\n",
              "      <td>Y</td>\n",
              "      <td>Y</td>\n",
              "      <td>0</td>\n",
              "      <td>225000.00000</td>\n",
              "      <td>Working</td>\n",
              "      <td>Higher education</td>\n",
              "      <td>Married</td>\n",
              "      <td>House / apartment</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>45.482192</td>\n",
              "      <td>7.832877</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1548 rows × 19 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7154fd0e-ed50-416b-b831-cb92be611aac')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-7154fd0e-ed50-416b-b831-cb92be611aac button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-7154fd0e-ed50-416b-b831-cb92be611aac');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-0cd80c76-6705-44dc-bb47-e2c9b7be54c2\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-0cd80c76-6705-44dc-bb47-e2c9b7be54c2')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-0cd80c76-6705-44dc-bb47-e2c9b7be54c2 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "  <div id=\"id_8b10516a-80c9-470c-b473-400064490602\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('credit_card_raw')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_8b10516a-80c9-470c-b473-400064490602 button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('credit_card_raw');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 169
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Removing Missing values"
      ],
      "metadata": {
        "id": "g0QHcZc7x-MT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw['GENDER']=credit_card_raw['GENDER'].fillna(credit_card_raw['GENDER'].mode()[0])\n",
        "credit_card_raw['Type_Occupation']=credit_card_raw['Type_Occupation'].fillna(credit_card_raw['Type_Occupation'].mode()[0])"
      ],
      "metadata": {
        "id": "M0BYqlo-mTzv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.isnull().sum()\n"
      ],
      "metadata": {
        "id": "Mo5TsJ8WmMoJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "02042bc2-ac6f-4a59-fa75-176bc0361ca6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Ind_ID             0\n",
              "GENDER             0\n",
              "Car_Owner          0\n",
              "Propert_Owner      0\n",
              "CHILDREN           0\n",
              "Annual_income      0\n",
              "Type_Income        0\n",
              "EDUCATION          0\n",
              "Marital_status     0\n",
              "Housing_type       0\n",
              "Mobile_phone       0\n",
              "Work_Phone         0\n",
              "Phone              0\n",
              "EMAIL_ID           0\n",
              "Type_Occupation    0\n",
              "Family_Members     0\n",
              "label              0\n",
              "age                0\n",
              "experience         0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 171
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# EDA"
      ],
      "metadata": {
        "id": "ZHwQdvpCzA2o"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Categorical variables visualization\n",
        "\n",
        "print(credit_card_raw['label'].value_counts())\n",
        "sns.countplot(x=credit_card_raw['label'])\n",
        "plt.title(\"Total Number of credit card approved and not approved people\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 541
        },
        "id": "-uYW1QPnzH-Q",
        "outputId": "fc9655b8-0a27-43d4-dd3e-a430170e64aa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0    1373\n",
            "1     175\n",
            "Name: label, dtype: int64\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Total Number of credit card approved and not approved people')"
            ]
          },
          "metadata": {},
          "execution_count": 172
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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rho1aphOnTpZzwu6vy2s9XmtsWdlZZmHHnrIlC1b1gwZMsTMnz/fxMTEGDc3N9OlSxeneUZHRxtJpnv37mbOnDmmd+/eRpLp2rWrU7+wsDBTp04d4+fnZ0aOHGmmTZtmGjdubFxcXMy6devyvO+X708L+v4UBEEqV5BatWqVkWRee+01p37du3c3DofD/Pjjj1ZbuXLl8t2RXf4ByxEfH28kmSVLllhtdoLUoUOHjJubmxk8eLA1vTCC1MCBA622S5cumSpVqhiHw2EmTpxotZ89e9Z4eXk5jTlnDHfccYdJS0uz2t99910jycyYMcMY83tArV27tomKijLZ2dlO71X16tXNgw8+mKemJ5544qrvS44hQ4YYSWbr1q1W26+//mqqV69uqlWrZrKyspzGP2jQoGvOc+PGjUaS0/uc4/L6JRkXFxezb98+pz79+/c3lStXdvoCMsaYHj16GF9fX2sbiYuLM5LMu+++a/VJT083tWrVumqQMub30JN7fV7NmDFjjCQrXOY3pkuXLpmMjAynaWfPnjVBQUGmX79+VlvONubj42OSk5Od+nft2tV4enqao0ePWm3fffedcXV1LdCOOb/PT2xsrHE4HE7zzNnZPv/8807j6NSpk3F3dzenTp1yqtXLy8v873//s/pu377dSDJDhw7NM8+RI0c6Lb+g+4UDBw4YSU5h2xhjnnvuOePt7W2NbevWrUaSWbZsmVO/NWvWOLUnJycbd3d306lTJ6ft7uWXXzaSChSkcr+fmZmZplGjRuaBBx5wapdk3N3dnfZxe/bsyTOeG12/bdq0ybMvzMjIMMHBwaZbt25WW85n45///KdT7REREcbb29va31zv5+DChQtO+wRjft9GPDw8zPjx4622gu7bLh/TG2+84TSmZs2amcDAQCsY58yzRo0aTuslMzPTBAYGmkaNGpnffvvNal+9erWRZMaMGWO1RUREmPDwcKf6d+zY4fSeXs/+trDW57XGvnTpUuPi4uK0nzbGmHnz5hlJZtu2bcYYY3bv3m0kmaefftqp34svvmgkmY0bN1ptYWFhRpJ57733rLbU1FRTuXJlc+edd1ptub9rr+f9KQhO7eXyySefyNXVVYMHD3Zqf+GFF2SM0aeffnrNeXh5eVn/vnjxon755RfVqlVLfn5++uqrr26ovho1aqhXr15asGCBfv755xua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bbWvX78+3/7Z2dn66aefrP+BStIPP/wgSXnuAp9fTT/88EOB7hZ/PfuF6tWrq2XLllq+fLliYmL0/vvvq2vXrvLw8LD61KxZU5999pnuueeeq36p5sz34MGDTkfeTp06le+R4dzee+89eXp6au3atU7Lf/vtt6/52vzc6Pq9HmFhYdq7d6+ys7Odgm3u9/x6PwcrV65U27Zt9Y9//MOpPSUlJc8ReOna+7YcJ0+ezHOrjCtti7nljOXAgQPWqa4cBw4cyLMPe/zxx/Xcc8/pwIEDWr58ucqWLavOnTtb0wu6vy2s9VmQsdesWVN79uxRu3btrrrOwsLClJ2drYMHDzrdzzEpKUkpKSl53osff/wxz/7zWu97YX8fcWovl44dOyorK0uzZ892ap8+fbocDoc6dOhgtZUrVy7fcOTq6prnf1azZs266hGh61WzZk099dRTmj9/fp7zwD4+PqpUqZK2bNni1P7mm28W2vJzW7JkiX799Vfr+cqVK/Xzzz9b71d4eLhq1qypqVOn6ty5c3lef+rUKdvL7tixo3bs2KH4+HirLT09XQsWLFC1atWcrl8qqG7duskYY93o73LX+l+zq6urunXrpvfeey/f4HL5WDt27KiTJ086/Rz9/PnzWrBgwTVrLFu2rKS8gflKunXrpj179uiDDz7IMy1nTDlHCi4f4/bt253e26txdXVVVFSUVq1apWPHjlnt+/fv19q1awv0+tzLN8ZoxowZV3zN5Z9VY4xmz56tMmXKqF27dk79Vq1a5fRT8h07dmj79u1On+kruZ79gvT7F92XX36phQsX6vTp006n9STpscceU1ZWliZMmJBnWZcuXbLWaWRkpMqUKaNZs2Y5vScF/dNGrq6ucjgcTvueI0eO2P6LDDe6fq9Hx44dlZiY6HS92aVLlzRr1ix5e3tbp3uv93OQ3/55xYoV+f7VCOna+7bLa7v8tgGZmZmaP3++AgICFB4eftWamjdvrsDAQM2bN8/pKNGnn36q/fv3q1OnTk79u3XrJldXV/3rX//SihUr9PDDDzuFmILubwtrfRZk7I899phOnDiht956K8/rf/vtN6Wnp0v6fb1LebfxnCNZud+LkydPOu3T0tLStGTJEjVr1kzBwcH51lvY30cckcqlc+fOatu2rf7617/qyJEjatq0qdatW6f//Oc/GjJkiNOFmOHh4frss880bdo0hYSEqHr16mrVqpUefvhhLV26VL6+vmrQoIHi4+P12WefqWLFioVa61//+lctXbpUBw4cUMOGDZ2mPf3005o4caKefvppNW/eXFu2bLFS+s3g7++ve++9V3379lVSUpLi4uJUq1YtDRgwQNLv17L8/e9/V4cOHdSwYUP17dtXd9xxh06cOKFNmzbJx8dHH330ka1ljxw5Uv/617/UoUMHDR48WP7+/lq8eLEOHz6s9957z9ZpmrZt26pXr16aOXOmDh48qPbt2ys7O1tbt25V27ZtnS5wzs/EiRO1adMmtWrVSgMGDFCDBg105swZffXVV/rss8905swZSdKAAQM0e/Zs9e7dWwkJCapcubKWLl1qfTlcjZeXlxo0aKDly5erTp068vf3V6NGja54HdTw4cO1cuVKPfroo+rXr5/Cw8N15swZffjhh5o3b56aNm2qhx9+WO+//77++Mc/qlOnTjp8+LDmzZunBg0a5LvDyc+4ceO0Zs0atW7dWs8995z15dewYUPt3bv3qq+tV6+eatasqRdffFEnTpyQj4+P3nvvvSseffH09NSaNWsUHR2tVq1a6dNPP9XHH3+sl19+Oc/psVq1aunee+/Vs88+q4yMDMXFxalixYp66aWXrjmm69kvSL9/abz44ot68cUX5e/vn+d/vW3atNEzzzyj2NhY7d69Ww899JDKlCmjgwcPasWKFZoxY4a6d++ugIAAvfjii4qNjdXDDz+sjh076uuvv9ann36a79GT3Dp16qRp06apffv2evLJJ5WcnKw5c+aoVq1a11wXV3Ij6/d6DBw4UPPnz1efPn2UkJCgatWqaeXKldq2bZvi4uKsI4PX+zl4+OGHNX78ePXt21d33323vvnmGy1btizPtXY5rrVvyxESEqJJkybpyJEjqlOnjpYvX67du3drwYIF17yZbpkyZTRp0iT17dtXbdq00RNPPKGkpCTNmDFD1apV09ChQ536BwYGqm3btpo2bZp+/fXXPEH9eva3hbE+CzL2Xr166d1339Wf//xnbdq0Sffcc4+ysrL0/fff691339XatWvVvHlzNW3aVNHR0VqwYIFSUlLUpk0b7dixQ4sXL1bXrl3z3KetTp066t+/v3bu3KmgoCAtXLhQSUlJVz3qWujfR9f1G79SKPftD4z5/afzQ4cONSEhIaZMmTKmdu3aZsqUKU4/kzTGmO+//97cd999xsvLy+mnyGfPnjV9+/Y1lSpVMt7e3iYqKsp8//33JiwsLN9bB1zP7Q9yy/m59uW3PzDm959x9u/f3/j6+pry5cubxx57zCQnJ1/x9gc5PxW/fL7lypXLs7zct1rIGcO//vUvM2rUKBMYGGi8vLxMp06dnH5Om+Prr782f/rTn0zFihWNh4eHCQsLM4899pjZsGHDNWu6mkOHDpnu3bsbPz8/4+npaVq2bJnnPjnGFPz2B8b8fiuAKVOmmHr16hl3d3cTEBBgOnToYBISEgo0v6SkJDNo0CATGhpqypQpY4KDg027du3MggULnPodPXrUPPLII6Zs2bKmUqVK5i9/+Yv1M/ir3f7AGGO++OILEx4ebtzd3Qv0E/BffvnFxMTEmDvuuMO4u7ubKlWqmOjoaOs2DdnZ2eb11183YWFhxsPDw9x5551m9erVeZadc0uBKVOm5LuczZs3W3XVqFHDzJs3z1qv1/Ldd9+ZyMhI4+3tbSpVqmQGDBhg/Qz/8lt65Gyjhw4dsu5PExQUZMaOHev08/bLa33jjTdMaGiodQ+43PcbutJ2b0zB9ws5cu67lvtn3JdbsGCBCQ8PN15eXqZ8+fKmcePG5qWXXjInT560+mRlZZlx48aZypUrGy8vL3P//febb7/9Ns/+5Er+8Y9/mNq1axsPDw9Tr1498/bbb+e7Lq60Lee3nBtZv7n3ITny276TkpKsfam7u7tp3Lhxvrd1uZ7PwYULF8wLL7xgvZ/33HOPiY+PN23atDFt2rSx+l3Pvi1nTLt27TIRERHG09PThIWFmdmzZzv1y5nn5fcTu9zy5cvNnXfeaTw8PIy/v7/p2bOn0y07LvfWW28ZSaZ8+fJOt0y4XEH2t8YUzvq81tiN+f02D5MmTTINGzY0Hh4epkKFCiY8PNyMGzfOpKamWv0uXrxoxo0bZ6pXr27KlCljQkNDzahRo8yFCxec5hcWFmY6depk1q5da5o0aWJt47nf3yt91xb0/bkWhzG34Oo+AChkffr00cqVK695pOzIkSOqXr26pkyZUiL/xiKKxueff662bdtqxYoVTrcnyc/999+v06dPX/MaxNKoKMderVo1NWrUSKtXr77ly74c10gBAADYRJACAACwiSAFAABgE9dIAQAA2MQRKQAAAJsIUgAAADZxQ84CyM7O1smTJ1W+fPmb+vd/AABA4THG6Ndff1VISMgN/Q3NqyFIFcDJkyfz/EFFAABQMhw/flxVqlS5KfMmSBVAzp8iOH78uHx8fIq4GgAAUBBpaWkKDQ11+mPjhY0gVQA5p/N8fHwIUgAAlDA387IcLjYHAACwiSAFAABgE0EKAADAJoIUAACATUUapLZs2aLOnTsrJCREDodDq1atumLfP//5z3I4HIqLi3NqP3PmjHr27CkfHx/5+fmpf//+OnfunFOfvXv3qnXr1vL09FRoaKgmT558E0YDAABuN0UapNLT09W0aVPNmTPnqv0++OADffnllwoJCckzrWfPntq3b5/Wr1+v1atXa8uWLRo4cKA1PS0tTQ899JDCwsKUkJCgKVOm6NVXX9WCBQsKfTwAAOD2UqS3P+jQoYM6dOhw1T4nTpzQ888/r7Vr16pTp05O0/bv3681a9Zo586dat68uSRp1qxZ6tixo6ZOnaqQkBAtW7ZMmZmZWrhwodzd3dWwYUPt3r1b06ZNcwpcAAAA16tYXyOVnZ2tXr16afjw4WrYsGGe6fHx8fLz87NClCRFRkbKxcVF27dvt/rcd999cnd3t/pERUXpwIEDOnv2bL7LzcjIUFpamtMDAAAgt2IdpCZNmiQ3NzcNHjw43+mJiYkKDAx0anNzc5O/v78SExOtPkFBQU59cp7n9MktNjZWvr6+1oM/DwMAAPJTbINUQkKCZsyYoUWLFt3yPxQ8atQopaamWo/jx4/f0uUDAICSodgGqa1btyo5OVlVq1aVm5ub3NzcdPToUb3wwguqVq2aJCk4OFjJyclOr7t06ZLOnDmj4OBgq09SUpJTn5znOX1y8/DwsP4cDH8WBgAAXEmxDVK9evXS3r17tXv3busREhKi4cOHa+3atZKkiIgIpaSkKCEhwXrdxo0blZ2drVatWll9tmzZoosXL1p91q9fr7p166pChQq3dlAAAKBUKdJf7Z07d04//vij9fzw4cPavXu3/P39VbVqVVWsWNGpf5kyZRQcHKy6detKkurXr6/27dtrwIABmjdvni5evKiYmBj16NHDulXCk08+qXHjxql///4aMWKEvv32W82YMUPTp0+/dQMFAAClUpEGqV27dqlt27bW82HDhkmSoqOjtWjRogLNY9myZYqJiVG7du3k4uKibt26aebMmdZ0X19frVu3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8mrTpo0k6dixY079IiIirH/7+/urbt262r9/vyRpz549WrRokby9va1HVFSUsrOzdfjw4Vs3GADFGkekAJQq6enpioqKUlRUlJYtW6aAgAAdO3ZMUVFRyszMLPB8zp07p2eeeUaDBw/OM61q1aqFWTKAEowgBaBU+f777/XLL79o4sSJCg0NlSTt2rUr375ffvmlFYrOnj2rH374QfXr15ck/eEPf9B3332nWrVq3ZrCAZRInNoDUKpUrVpV7u7umjVrln766Sd9+OGHmjBhQr59x48frw0bNujbb79Vnz59VKlSJXXt2lWSNGLECH3xxReKiYnR7t27dfDgQf3nP//hYnMATghSAEqVgIAALVq0SCtWrFCDBg00ceJETZ06Nd++EydO1F/+8heFh4crMTFRH330kdzd3SVJTZo00ebNm/XDDz+odevWuvPOOzVmzBiFhITcyuEAKOYcxhhT1EUAAACURByRAgAAsIkgBQAAYBNBCgAAwCaCFAAAgE0EKQAAAJsIUgAAADYRpAAAAGwiSAEAANhEkAIAALCJIAUAAGATQQoAAMAmghQAAIBN/w8YuCZ/VjLflQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(x=credit_card_raw['GENDER'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "a77zyw-0zICD",
        "outputId": "55fbd381-4db8-4b21-b84f-33e9a0a0b9cc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='GENDER', ylabel='count'>"
            ]
          },
          "metadata": {},
          "execution_count": 173
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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3rzmWh8AAECLa9IVov3798vr9TYaT05O5i0zAABw3GlSEPn9fk2bNk21tbX22Pfff697771Xfr+/2RYHAADQGpr0ltmjjz6qoUOHqnv37jrzzDMlSR999JHcbreWLVvWrAsEAABoaU0Kon79+mnLli1auHChNm/eLEkaNWqUsrKyFBcX16wLBAAAaGlNCqLCwkJ5vV6NGzcubPyZZ57Rzp07NWXKlGZZHAAAQGto0j1ETz75pPr06dNovG/fvioqKvrViwIAAGhNTQqiQCCgrl27Nhrv0qWLduzY8asXBQAA0JqaFESpqalas2ZNo/E1a9YoJSXlVy8KAACgNTXpHqJx48Zp4sSJOnDggC655BJJUnFxsSZPnsw3VQMAgONOk4Jo0qRJ+u6773TzzTervr5ekhQbG6spU6YoPz+/WRcIAADQ0poURC6XSw899JCmTp2qTZs2KS4uTqeeeqrcbndzrw8AAKDFNSmIDktISNA555zTXGsBAACIiCbdVA0AAHAiIYgAAIDxCCIAAGA8gggAABiPIAIAAMYjiAAAgPEIIgAAYDyCCAAAGI8gAgAAxiOIAACA8QgiAABgPIIIAAAYjyACAADGI4gAAIDxCCIAAGC84yqIHnzwQblcLk2cONEeq62tVU5Ojjp16qSEhASNGDFCVVVVYa+rrKzUsGHD1K5dOyUnJ2vSpEk6ePBgK68eAAA41XETROvWrdOTTz6pM844I2z89ttv1+uvv66XXnpJJSUl2r59u66++mp7+6FDhzRs2DDV19fr3Xff1XPPPacFCxaooKCgtU8BAAA41HERRHv37lVWVpaeeuopdejQwR4PBoN6+umnNWvWLF1yySUaMGCAnn32Wb377rt67733JEnLli3Tp59+queff179+/fXZZddpvvuu09z585VfX19pE4JAAA4yHERRDk5ORo2bJgyMjLCxsvLy3XgwIGw8T59+qhHjx4qLS2VJJWWlqpfv37yer32nMzMTIVCIW3cuPGIx6urq1MoFAp7AACAE1d0pBfwS1544QV98MEHWrduXaNtgUBAMTExSkpKChv3er0KBAL2nB/G0OHth7cdSWFhoe69995mWD0AADgeOPoK0bZt23Tbbbdp4cKFio2NbbXj5ufnKxgM2o9t27a12rEBAEDrc3QQlZeXq7q6Wmeffbaio6MVHR2tkpISzZkz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WpHLsiwr0osAAACIJD52DwAAjEcQAQAA4xFEAADAeAQRAAAwHkEEAACMRxABAADjEUQAAMB4BBEAADAeQQQAAIxHEAFwrEAgoNtuu029evVSbGysvF6vBg8erHnz5mn//v2SpJNOOkkul6vR48EHH5QkffXVV3K5XEpOTtaePXvC9t+/f3/dc8899vOLLrrIfr3b7Va3bt105ZVX6pVXXmm0tiMd0+Vy6YUXXpAkvf3222HjXbp00eWXX64NGza00L8WgF+DIALgSF9++aXOOussLVu2TA888IA+/PBDlZaWavLkyVqyZIn+/e9/23OnT5+uHTt2hD1uueWWsP3t2bNHf/nLX37xuOPGjdOOHTv0xRdf6OWXX1Z6erpGjhyp8ePHN5r77LPPNjru8OHDw+ZUVFRox44deuutt1RXV6dhw4apvr6+af8oAFoMf9wVgCPdfPPNio6O1vvvv6/4+Hh7/OSTT9bvf/97/fDPMLZv314+n+9n93fLLbdo1qxZysnJUXJy8k/Oa9eunb2v7t2767zzzlOfPn00ZswYXXvttcrIyLDnJiUl/eJxk5OT7XkTJ07UVVddpc2bN+uMM8742dcBaF1cIQLgON99952WLVumnJycsBj6IZfLdUz7HDVqlHr16qXp06cf83qys7PVoUOHI751drSCwaD9dlpMTEyT9wOgZRBEABzn888/l2VZ6t27d9h4586dlZCQoISEBE2ZMsUenzJlij1++LF69eqw1x6+r2j+/Pn64osvjmk9UVFR+s1vfqOvvvoqbHzUqFGNjltZWRk2p3v37kpISFBSUpIWLVqkq666Sn369Dmm4wNoebxlBuC4sXbtWjU0NCgrK0t1dXX2+KRJkzR69Oiwud26dWv0+szMTF1wwQWaOnWqFi1adEzHtiyr0VWp2bNnh72FJkkpKSlhz1evXq127drpvffe0wMPPKCioqJjOi6A1kEQAXCcXr16yeVyqaKiImz85JNPliTFxcWFjXfu3Fm9evU6qn0/+OCD8vv9mjRp0lGv59ChQ9qyZYvOOeecsHGfz/eLx01LS1NSUpJ69+6t6upqXXfddVq1atVRHxtA6+AtMwCO06lTJ/3ud7/T448/rn379jXrvs8991xdffXVuuuuu476Nc8995x2796tESNG/Kpj5+Tk6JNPPtHixYt/1X4AND+uEAFwpCeeeEKDBw/WwIEDdc899+iMM85QVFSU1q1bp82bN2vAgAH23D179igQCIS9vl27dvJ4PEfc94wZM9S3b19FRzf+X+D+/fsVCAR08OBBffPNN1q8eLFmz56tCRMm6OKLLw6bW1NT0+i47du3/8kbwdu1a6dx48Zp2rRpGj58+DHfGA6gBVkA4FDbt2+3cnNzrbS0NKtt27ZWQkKCde6551oPP/ywtW/fPsuyLKtnz56WpEaPm266ybIsy9q6daslyfrwww/D9j1+/HhLkjVt2jR77Le//a39+piYGKtr167WFVdcYb3yyiuN1nakY0qyCgsLLcuyrJUrV1qSrN27d4e9rrKy0oqOjrZefPHF5vuHAvCruSzrB1/mAQAAYCDuIQIAAMYjiAAAgPEIIgAAYDyCCAAAGI8gAgAAxiOIAACA8QgiAABgPIIIAAAYjyACAADGI4gAAIDxCCIAAGC8/wdYjTFN3oBRowAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,4))\n",
        "sns.countplot(x=credit_card_raw['Marital_status'])\n",
        "plt.title(\"Number of people Marital status wise\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 427
        },
        "id": "lcCRc5uqNMV2",
        "outputId": "a7600211-846e-459c-f609-d97cb1e88b12"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Number of people Marital status wise')"
            ]
          },
          "metadata": {},
          "execution_count": 174
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x400 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1sAAAGJCAYAAAB8VSkIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABTMElEQVR4nO3de3yP9f/H8ednw7Z2NHYw5nxmkTk0p6FpDgkJZWXOCknLsa9jJ0JCX+VQmURUpFIpOc6hOYTEQnLq6zA5bOa4w/X7w23Xz8eGmV02PO6322583tf7el+v6/pcn2t77jrMZhiGIQAAAABAjnLI7QIAAAAA4H5E2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgCLrV69WjabTV999VVul5IlJ06c0NNPP61ChQrJZrNp8uTJuV3SHenSpYtKliyZ22XkiIMHD8pmsyk6OjpHxy1ZsqS6dOmSo2M+SGw2m0aPHp3bZQDIgwhbAO4L0dHRstlscnZ21v/+978M0xs1aqSqVavmQmX3nldeeUU//fSThg0bprlz56pZs2a5XVKeMnr0aNlsNjk4OOjIkSMZpicmJsrFxUU2m039+vWzvJ4ffvghz/ygv3v3bo0ePVoHDx7M9hjz58+/5wM+AKQjbAG4r1y+fFnjxo3L7TLuaStXrlTr1q01cOBAPffcc6pYsWJul5QnOTk56fPPP8/QvnjxYsuWWaJECV28eFHPP/+82fbDDz9ozJgxli3zduzevVtjxox54MLWxYsXNXz48NwuA0AeRNgCcF+pXr26Zs2apaNHj+Z2KXfd+fPnc2Sc+Ph4eXl55chY97MWLVpkGrbmz5+vli1b5uiyUlJSdOXKFfPsraOjY46Ojzvj7OysfPny5XYZAPIgwhaA+8prr72m1NTUW57dutm9L9fff5F+2djevXv13HPPydPTUz4+PhoxYoQMw9CRI0fUunVreXh4yN/fX++++26my0xNTdVrr70mf39/ubq66sknn8z0MrTY2Fg1a9ZMnp6eeuihhxQaGqr169fb9Umvaffu3erUqZMKFiyo+vXr33Sd//77b7Vv317e3t566KGH9Oijj+r77783p6dfimkYhqZNmyabzSabzXbLbThx4kS99957KlGihFxcXBQaGqo//vgjQ/8///xTTz/9tLy9veXs7KyaNWvq22+/ve06pf+/D27hwoVZ2qbXS0tL0+TJk1WlShU5OzvLz89PvXv31pkzZ245b7pOnTpp+/bt+vPPP82248ePa+XKlerUqVOG/leuXNHIkSMVHBwsT09Pubq6qkGDBlq1apVdv2u36+TJk1WmTBk5OTlp9+7dGfbbLl26aNq0aZJkvl/XvmcTJ05U3bp1VahQIbm4uCg4OPiO7h1csGCBgoOD5e7uLg8PDwUFBWnKlCmSru4/7du3lyQ1btzYrGX16tWSpG+++UYtW7ZUQECAnJycVKZMGb3xxhtKTU01x2/UqJG+//57HTp0yJw//X679P3z+rNm6ftC+nIkad++fWrXrp38/f3l7OysYsWK6ZlnnlFCQsIN123q1KlydHTU2bNnzbZ3331XNptNUVFRZltqaqrc3d01ZMgQs+36Y8a5c+c0YMAAlSxZUk5OTvL19VXTpk3122+/2S0zK591APc2fg0D4L5SqlQpde7cWbNmzdLQoUMVEBCQY2N37NhRlSpV0rhx4/T999/rzTfflLe3t2bMmKEmTZronXfe0bx58zRw4EDVqlVLDRs2tJv/rbfeks1m05AhQxQfH6/JkycrLCxM27dvl4uLi6Srl/A1b95cwcHBGjVqlBwcHDR79mw1adJEMTExql27tt2Y7du3V7ly5fT222/LMIwb1n7ixAnVrVtXFy5cUP/+/VWoUCHNmTNHTz75pL766iu1bdtWDRs21Ny5c/X888+radOm6ty5c5a2y6effqpz586pb9++unTpkqZMmaImTZpo586d8vPzkyTt2rVL9erVU9GiRTV06FC5urrqiy++UJs2bbRo0SK1bds2y3Xe7jbNTO/evRUdHa2uXbuqf//+OnDggP773/9q27ZtWr9+vfLnz3/L9W7YsKGKFSum+fPn6/XXX5ckLVy4UG5ubpme2UpMTNRHH32kZ599Vj179tS5c+f08ccfKzw8XJs2bVL16tXt+s+ePVuXLl1Sr1695OTkJG9vb6WlpWVYj6NHj2r58uWaO3duhmVOmTJFTz75pCIiInTlyhUtWLBA7du319KlS2/77Nvy5cv17LPP6rHHHtM777wjSYqLi9P69ev18ssvq2HDhurfv7+mTp2q1157TZUqVZIk89/o6Gi5ubkpKipKbm5uWrly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tgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAvk+bD1v//9T88995wKFSokFxcXBQUFacuWLeZ0wzA0cuRIFSlSRC4uLgoLC9O+ffvsxjh9+rQiIiLk4eEhLy8vde/eXUlJSXd7VQAAAAA8QLIVtpo0aaKzZ89maE9MTFSTJk3utCbTmTNnVK9ePeXPn18//vijdu/erXfffVcFCxY0+4wfP15Tp07V9OnTFRsbK1dXV4WHh+vSpUtmn4iICO3atUvLly/X0qVLtXbtWvXq1SvH6gQAAACA69kMwzBudyYHBwcdP35cvr6+du3x8fEqWrSokpOTc6S4oUOHav369YqJicl0umEYCggI0KuvvqqBAwdKkhISEuTn56fo6Gg988wziouLU+XKlbV582bVrFlTkrRs2TK1aNFC//zzjwICAm5ZR2Jiojw9PZWQkCAPD49srUvwoE+zNR/uTVsndM7tEgAAAGCB28kGt3Vm6/fff9fvv/8uSdq9e7f5+vfff9e2bdv08ccfq2jRotmv/Drffvutatasqfbt28vX11ePPPKIZs2aZU4/cOCAjh8/rrCwMLPN09NTderU0caNGyVJGzdulJeXlxm0JCksLEwODg6KjY3NdLmXL19WYmKi3RcAAAAA3I58t9O5evXqstlsstlsmV4u6OLiovfffz/Hivv777/14YcfKioqSq+99po2b96s/v37q0CBAoqMjNTx48clSX5+fnbz+fn5mdMyOwOXL18+eXt7m32uN3bsWI0ZMybH1gMAAADAg+e2wtaBAwdkGIZKly6tTZs2ycfHx5xWoEAB+fr6ytHRMceKS0tLU82aNfX2229Lkh555BH98ccfmj59uiIjI3NsOdcbNmyYoqKizNeJiYkKDAy0bHkAAAAA7j+3FbZKlCgh6WoIuhuKFCmiypUr27VVqlRJixYtkiT5+/tLkk6cOKEiRYqYfU6cOKHq1aubfeLj4+3GSElJ0enTp835r+fk5CQnJ6ecWg0AAAAAD6DbClvX2rdvn1atWqX4+PgM4WvkyJF3XJgk1atXT3v27LFr27t3rxn6SpUqJX9/f61YscIMV4mJiYqNjdWLL74oSQoJCdHZs2e1detWBQcHS5JWrlyptLQ01alTJ0fqBAAAAIDrZStszZo1Sy+++KIKFy4sf39/2Ww2c5rNZsuxsPXKK6+obt26evvtt9WhQwdt2rRJM2fO1MyZM81lDRgwQG+++abKlSunUqVKacSIEQoICFCbNm0kXT0T1qxZM/Xs2VPTp09XcnKy+vXrp2eeeSZLTyIEAAAAgOzIVth688039dZbb2nIkCE5XY+dWrVq6euvv9awYcP0+uuvq1SpUpo8ebIiIiLMPoMHD9b58+fVq1cvnT17VvXr19eyZcvk7Oxs9pk3b5769eunxx57TA4ODmrXrp2mTp1qae0AAAAAHmzZ+jtbHh4e2r59u0qXLm1FTXkOf2cLt4u/swUAAHB/suzvbKVr3769fv7552wVBwAAAAAPgmxdRli2bFmNGDFCv/76q4KCgpQ/f3676f3798+R4gAAAADgXpWtsDVz5ky5ublpzZo1WrNmjd00m81G2AIAAADwwMtW2Dpw4EBO1wEAAAAA95Vs3bMFAAAAALi5bJ3Z6tat202nf/LJJ9kqBgAAAADuF9kKW2fOnLF7nZycrD/++ENnz55VkyZNcqQwAAAAALiXZStsff311xna0tLS9OKLL6pMmTJ3XBQAAAAA3Oty7J4tBwcHRUVF6b333supIQEAAADgnpWjD8jYv3+/UlJScnJIAAAAALgnZesywqioKLvXhmHo2LFj+v777xUZGZkjhQEAAADAvSxbYWvbtm12rx0cHOTj46N33333lk8qBAAAAIAHQbbC1qpVq3K6DgAAAAC4r2QrbKU7efKk9uzZI0mqUKGCfHx8cqQoAAAAALjXZesBGefPn1e3bt1UpEgRNWzYUA0bNlRAQIC6d++uCxcu5HSNAAAAAHDPyVbYioqK0po1a/Tdd9/p7NmzOnv2rL755hutWbNGr776ak7XCAAAAAD3nGxdRrho0SJ99dVXatSokdnWokULubi4qEOHDvrwww9zqj4AAAAAuCdl68zWhQsX5Ofnl6Hd19eXywgBAAAAQNkMWyEhIRo1apQuXbpktl28eFFjxoxRSEhIjhUHAAAAAPeqbF1GOHnyZDVr1kzFihVTtWrVJEk7duyQk5OTfv755xwtEAAAAADuRdkKW0FBQdq3b5/mzZunP//8U5L07LPPKiIiQi4uLjlaIAAAAADci7IVtsaOHSs/Pz/17NnTrv2TTz7RyZMnNWTIkBwpDgAAAADuVdm6Z2vGjBmqWLFihvYqVapo+vTpd1wUAAAAANzrshW2jh8/riJFimRo9/Hx0bFjx+64KAAAAAC412UrbAUGBmr9+vUZ2tevX6+AgIA7LgoAAAAA7nXZumerZ8+eGjBggJKTk9WkSRNJ0ooVKzR48GC9+uqrOVogAAAAANyLshW2Bg0apFOnTqlPnz66cuWKJMnZ2VlDhgzRsGHDcrRAAAAAALgXZSts2Ww2vfPOOxoxYoTi4uLk4uKicuXKycnJKafrAwAAAIB7UrbCVjo3NzfVqlUrp2oBAAAAgPtGth6QAQAAAAC4OcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAF7qmwNW7cONlsNg0YMMBsu3Tpkvr27atChQrJzc1N7dq104kTJ+zmO3z4sFq2bKmHHnpIvr6+GjRokFJSUu5y9QAAAAAeJPdM2Nq8ebNmzJihhx9+2K79lVde0Xfffacvv/xSa9as0dGjR/XUU0+Z01NTU9WyZUtduXJFGzZs0Jw5cxQdHa2RI0fe7VUAAAAA8AC5J8JWUlKSIiIiNGvWLBUsWNBsT0hI0Mcff6xJkyapSZMmCg4O1uzZs7Vhwwb9+uuvkqSff/5Zu3fv1meffabq1aurefPmeuONNzRt2jRduXIlt1YJAAAAwH3unghbffv2VcuWLRUWFmbXvnXrViUnJ9u1V6xYUcWLF9fGjRslSRs3blRQUJD8/PzMPuHh4UpMTNSuXbsyXd7ly5eVmJho9wUAAAAAtyNfbhdwKwsWLNBvv/2mzZs3Z5h2/PhxFShQQF5eXnbtfn5+On78uNnn2qCVPj19WmbGjh2rMWPG5ED1AAAAAB5UefrM1pEjR/Tyyy9r3rx5cnZ2vmvLHTZsmBISEsyvI0eO3LVlAwAAALg/5OmwtXXrVsXHx6tGjRrKly+f8uXLpzVr1mjq1KnKly+f/Pz8dOXKFZ09e9ZuvhMnTsjf31+S5O/vn+HphOmv0/tcz8nJSR4eHnZfAAAAAHA78nTYeuyxx7Rz505t377d/KpZs6YiIiLM/+fPn18rVqww59mzZ48OHz6skJAQSVJISIh27typ+Ph4s8/y5cvl4eGhypUr3/V1AgAAAPBgyNP3bLm7u6tq1ap2ba6uripUqJDZ3r17d0VFRcnb21seHh566aWXFBISokcffVSS9Pjjj6ty5cp6/vnnNX78eB0/flzDhw9X37595eTkdNfXCQAAAMCDIU+Hrax477335ODgoHbt2uny5csKDw/XBx98YE53dHTU0qVL9eKLLyokJESurq6KjIzU66+/notVAwAAALjf2QzDMHK7iLwuMTFRnp6eSkhIyPb9W8GDPs3hqpCXbZ3QObdLAAAAgAVuJxvk6Xu2AAAAAOBeRdgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALJAvtwsAkLMOvx6U2yXgLio+cmdulwAAAG6AM1sAAAAAYAHCFgAAAABYgLAFAAAAABbI02Fr7NixqlWrltzd3eXr66s2bdpoz549dn0uXbqkvn37qlChQnJzc1O7du104sQJuz6HDx9Wy5Yt9dBDD8nX11eDBg1SSkrK3VwVAAAAAA+YPB221qxZo759++rXX3/V8uXLlZycrMcff1znz583+7zyyiv67rvv9OWXX2rNmjU6evSonnrqKXN6amqqWrZsqStXrmjDhg2aM2eOoqOjNXLkyNxYJQAAAAAPCJthGEZuF5FVJ0+elK+vr9asWaOGDRsqISFBPj4+mj9/vp5++mlJ0p9//qlKlSpp48aNevTRR/Xjjz/qiSee0NGjR+Xn5ydJmj59uoYMGaKTJ0+qQIECt1xuYmKiPD09lZCQIA8Pj2zVHjzo02zNh3vT1gmdc23ZPI3wwcLTCAEAuLtuJxvk6TNb10tISJAkeXt7S5K2bt2q5ORkhYWFmX0qVqyo4sWLa+PGjZKkjRs3KigoyAxakhQeHq7ExETt2rUr0+VcvnxZiYmJdl8AAAAAcDvumbCVlpamAQMGqF69eqpataok6fjx4ypQoIC8vLzs+vr5+en48eNmn2uDVvr09GmZGTt2rDw9Pc2vwMDAHF4bAAAAAPe7eyZs9e3bV3/88YcWLFhg+bKGDRumhIQE8+vIkSOWLxMAAADA/SVfbheQFf369dPSpUu1du1aFStWzGz39/fXlStXdPbsWbuzWydOnJC/v7/ZZ9OmTXbjpT+tML3P9ZycnOTk5JTDawEAAADgQZKnz2wZhqF+/frp66+/1sqVK1WqVCm76cHBwcqfP79WrFhhtu3Zs0eHDx9WSEiIJCkkJEQ7d+5UfHy82Wf58uXy8PBQ5cqV786KAAAAAHjg5OkzW3379tX8+fP1zTffyN3d3bzHytPTUy4uLvL09FT37t0VFRUlb29veXh46KWXXlJISIgeffRRSdLjjz+uypUr6/nnn9f48eN1/PhxDR8+XH379uXsFQAAAADL5Omw9eGHH0qSGjVqZNc+e/ZsdenSRZL03nvvycHBQe3atdPly5cVHh6uDz74wOzr6OiopUuX6sUXX1RISIhcXV0VGRmp119//W6tBgAAAIAHUJ4OW1n5E2DOzs6aNm2apk2bdsM+JUqU0A8//JCTpQEAAADATeXpe7YAAAAA4F5F2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALBAvtwuAABwb6r3fr3cLgF30fqX1ud2CQBwz+HMFgAAAABYgDNbAAAgT1vTMDS3S8BdFLp2TW6XAOQYzmwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFeBohAAAAIOm/r36X2yXgLur3bivLl8GZLQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALAAYQsAAAAALEDYAgAAAAALELYAAAAAwAKELQAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMACD1TYmjZtmkqWLClnZ2fVqVNHmzZtyu2SAAAAANynHpiwtXDhQkVFRWnUqFH67bffVK1aNYWHhys+Pj63SwMAAABwH3pgwtakSZPUs2dPde3aVZUrV9b06dP10EMP6ZNPPsnt0gAAAADch/LldgF3w5UrV7R161YNGzbMbHNwcFBYWJg2btyYof/ly5d1+fJl83VCQoIkKTExMds1pF6+mO15ce+5k33lTp27lJpry8bdl5v7WsrFlFxbNu6+3NzXzqewrz1IcnNfu3j5Qq4tG3dfdve19PkMw7hl3wcibP37779KTU2Vn5+fXbufn5/+/PPPDP3Hjh2rMWPGZGgPDAy0rEbcXzzffyG3S8CDYqxnbleAB4TnEPY13CWe7Gu4OwZPu7P5z507J89b7K8PRNi6XcOGDVNUVJT5Oi0tTadPn1ahQoVks9lysbJ7S2JiogIDA3XkyBF5eHjkdjm4j7Gv4W5hX8Pdwr6Gu4V97fYZhqFz584pICDgln0fiLBVuHBhOTo66sSJE3btJ06ckL+/f4b+Tk5OcnJysmvz8vKyssT7moeHBx9e3BXsa7hb2Ndwt7Cv4W5hX7s9tzqjle6BeEBGgQIFFBwcrBUrVphtaWlpWrFihUJCQnKxMgAAAAD3qwfizJYkRUVFKTIyUjVr1lTt2rU1efJknT9/Xl27ds3t0gAAAADchx6YsNWxY0edPHlSI0eO1PHjx1W9enUtW7Ysw0MzkHOcnJw0atSoDJdkAjmNfQ13C/sa7hb2Ndwt7GvWshlZeWYhAAAAAOC2PBD3bAEAAADA3UbYAgAAAAALELYAAAAAwAKELeSKkiVLavLkyXc0xujRo1W9evUcqed+ZbPZtGTJkhwdk+1ujdWrV8tms+ns2bN3NE5OfLbyitvdf69fdyv2/5wWHR3N33FEntalSxe1adMmt8tANmXlewvHIWsRth5gXbp0kc1m0wsvvJBhWt++fWWz2dSlSxdLlr1582b16tXLkrEfFCdPntSLL76o4sWLy8nJSf7+/goPD9f69evNPseOHVPz5s1zscqb69q1q4YPH27Z+PdSMKxbt66OHTuW5T+SeK87fvy4XnrpJZUuXVpOTk4KDAxUq1at7P4e4u3uv/ficaVjx47au3dvbpeB62Tl+JqXEZDuT9OnT5e7u7tSUlLMtqSkJOXPn1+NGjWy65sesooUKfJAfW/Jix6YR78jc4GBgVqwYIHee+89ubi4SJIuXbqk+fPnq3jx4nc0dnJysvLnz2/XduXKFRUoUEA+Pj53NDakdu3a6cqVK5ozZ45Kly6tEydOaMWKFTp16pTZx9/fPxcrvLnU1FQtXbpU33//fW6Xctek7//XS05OVoECBfL0+5WTDh48qHr16snLy0sTJkxQUFCQkpOT9dNPP6lv3776888/Jd3+/ptXjys3e99dXFzMYy/yjqwcX3NDZt9X8eBo3LixkpKStGXLFj366KOSpJiYGPn7+ys2NlaXLl2Ss7OzJGnVqlUqXry4KlSokJslQ5zZeuDVqFFDgYGBWrx4sdm2ePFiFS9eXI888ojZtmzZMtWvX19eXl4qVKiQnnjiCe3fv9+cfvDgQdlsNi1cuFChoaFydnbWvHnzzN+uvfXWWwoICDA/9Ndf7nP27Fn16NFDPj4+8vDwUJMmTbRjxw67WseNGyc/Pz+5u7ure/fuunTpkkVbJe87e/asYmJi9M4776hx48YqUaKEateurWHDhunJJ580+117GVX6e7R48WI1btxYDz30kKpVq6aNGzfajT1r1iwFBgbqoYceUtu2bTVp0qRbXl7w0UcfqVKlSnJ2dlbFihX1wQcf3HIdNmzYoPz586tWrVqZTm/UqJH69++vwYMHy9vbW/7+/ho9erRdn8OHD6t169Zyc3OTh4eHOnTooBMnTki6elnEmDFjtGPHDtlsNtlsNkVHR2e6rPT99O2335afn5+8vLz0+uuvKyUlRYMGDZK3t7eKFSum2bNn2803ZMgQlS9fXg899JBKly6tESNGKDk52Zyefmbto48+UqlSpcxvgjabTR9++KGefPJJubq66q233sr0Uo9169apQYMGcnFxUWBgoPr376/z58+b0+Pj49WqVSu5uLioVKlSmjdv3i23e17Qp08f2Ww2bdq0Se3atVP58uVVpUoVRUVF6ddffzX7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1po1a7R37141aNBAjzzyiEaOHGn+llySZs+erYCAAIWGhuqpp55Sr1695Ovrm2M1WKV06dL67bff1LhxY7366quqWrWqmjZtqhUrVujDDz+86bwRERHasWOHGjRocMd/CzA7evfuraeeekodO3ZUnTp1dOrUKbszDHdq0qRJKliwoOrWratWrVopPDw8w1mFTz/9VH5+fmrYsKHatm2rnj17yt3d3e5PC9zqmIobu9XxNavbt2zZsnrqqafUokULPf7443r44YftnqL78ccf68yZM6pRo4aef/559e/fP0uf36zsgz179lSFChVUs2ZN+fj4aP369XrooYe0du1aFS9eXE899ZQqVapk/hmV9O/9r776qp5//nlFRkaalyimn2VF3tG4cWNdvHhRZcuWtdvnQkNDde7cOfMR8Zl59NFHNWvWLE2ZMkXVqlXTzz//rOHDh9v1adOmjaZMmaKJEyeqSpUqmjFjhmbPnm13tqpgwYIKCgqSj4+PKlasKOlqAEtLS+N+revYjOzerQzggdCzZ0/9+eefiomJyZHxnnzySdWvX1+DBw/OkfGAB90///yjwMBA/fLLL7d86AzujtGjR2vJkiV2f+4BwIOJe7YA2Jk4caKaNm0qV1dX/fjjj5ozZ06O/k2z+vXr58rlX8D9YuXKlUpKSlJQUJCOHTumwYMHq2TJkuZ9NgCAvIOwBcDOpk2bNH78eJ07d06lS5fW1KlT1aNHjxwbnzNawJ1JTk7Wa6+9pr///lvu7u6qW7eu5s2bl+HJigCA3MdlhAAAAABgAR6QAQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQDuO9HR0fLy8soz4wAAHkyELQDAXdWlSxfZbDa98MILGab17dtXNptNXbp0uaNldOzYUXv37jVfjx49WtWrV7+jMe9EdkPb6tWrZbPZdPbs2RyvCQBgPcIWAOCuCwwM1IIFC3Tx4kWz7dKlS5o/f76KFy9+R2MnJyfLxcVFvr6+d1omAAB3hLAFALjratSoocDAQC1evNhsW7x4sYoXL65HHnnEbFu2bJnq168vLy8vFSpUSE888YT2799vTj948KBsNpsWLlyo0NBQOTs7a968eXZnkqKjozVmzBjt2LFDNptNNptN0dHRkqRJkyYpKChIrq6uCgwMVJ8+fZSUlJStddqxY4caN24sd3d3eXh4KDg4WFu2bNHq1avVtWtXJSQkmMsfPXq0JGnu3LmqWbOm3N3d5e/vr06dOik+Pt5ct8aNG0uSChYsaHfGr2TJkpo8ebLd8qtXr26OaxiGRo8ereLFi8vJyUkBAQHq379/ttYLAJB9hC0AQK7o1q2bZs+ebb7+5JNP1LVrV7s+58+fV1RUlLZs2aIVK1bIwcFBbdu2VVpaml2/oUOH6uWXX1ZcXJzCw8PtpnXs2FGvvvqqqlSpomPHjunYsWPq2LGjJMnBwUFTp07Vrl27NGfOHK1cuVKDBw/O1vpERESoWLFi2rx5s7Zu3aqhQ4cqf/78qlu3riZPniwPDw9z+QMHDpR09SzcG2+8oR07dmjJkiU6ePCgGagCAwO1aNEiSdKePXt07NgxTZkyJUu1LFq0SO+9955mzJihffv2acmSJQoKCsrWegEAsi9fbhcAAHgwPffccxo2bJgOHTokSVq/fr0WLFig1atXm33atWtnN88nn3wiHx8f7d69W1WrVjXbBwwYoKeeeirT5bi4uMjNzU358uWTv7+/3bQBAwaY/y9ZsqTefPNNvfDCC/rggw9ue30OHz6sQYMGqWLFipKkcuXKmdM8PT1ls9kyLL9bt27m/0uXLq2pU6eqVq1aSkpKkpubm7y9vSVJvr6+t3XP1+HDh+Xv76+wsDDlz59fxYsXV+3atW97nQAAd4YzWwCAXOHj46OWLVsqOjpas2fPVsuWLVW4cGG7Pvv27dOzzz6r0qVLy8PDQyVLlpR0NUxcq2bNmtmq4ZdfftFjjz2mokWLyt3dXc8//7xOnTqlCxcu3PZYUVFR6tGjh8LCwjRu3Di7yx1vZOvWrWrVqpWKFy8ud3d3hYaGSsq4frerffv2unjxokqXLq2ePXvq66+/VkpKyh2NCQC4fYQtAECu6datm6KjozVnzhy7szzpWrVqpdOnT2vWrFmKjY1VbGysJOnKlSt2/VxdXW972QcPHtQTTzyhhx9+WIsWLdLWrVs1bdq0TMfPitGjR2vXrl1q2bKlVq5cqcqVK+vrr7++Yf/z588rPDxcHh4emjdvnjZv3mz2v9XyHRwcZBiGXVtycrL5/8DAQO3Zs0cffPCBXFxc1KdPHzVs2NCuDwDAelxGCADINc2aNdOVK1dks9ky3Gt16tQp7dmzR7NmzVKDBg0kSevWrcvWcgoUKKDU1FS7tq1btyotLU3vvvuuHByu/u7xiy++yNb46cqXL6/y5cvrlVde0bPPPqvZs2erbdu2mS7/zz//1KlTpzRu3DgFBgZKkrZs2ZKhbkkZ5vXx8dGxY8fM14mJiTpw4IBdHxcXF7Vq1UqtWrVS3759VbFiRe3cuVM1atS4o3UEAGQdZ7YAALnG0dFRcXFx2r17txwdHe2mFSxYUIUKFdLMmTP1119/aeXKlYqKisrWckqWLKkDBw5o+/bt+vfff3X58mWVLVtWycnJev/99/X3339r7ty5mj59erbGv3jxovr166fVq1fr0KFDWr9+vTZv3qxKlSqZy09KStKKFSv077//6sKFCypevLgKFChgLv/bb7/VG2+8YTduiRIlZLPZtHTpUp08edJ8UmKTJk00d+5cxcTEaOfOnYqMjLTbftHR0fr444/1xx9/6O+//9Znn30mFxcXlShRIlvrBwDIHsIWACBXeXh4yMPDI0O7g4ODFixYoK1bt6pq1ap65ZVXNGHChGwto127dmrWrJkaN24sHx8fff7556pWrZomTZqkd955R1WrVtW8efM0duzYbI3v6OioU6dOqXPnzipfvrw6dOig5s2ba8yYMZKkunXr6oUXXlDHjh3l4+Oj8ePHy8fHR9HR0fryyy9VuXJljRs3ThMnTrQbt2jRohozZoyGDh0qPz8/9evXT5I0bNgwhYaG6oknnlDLli3Vpk0blSlTxpzPy8tLs2bNUr169fTwww/rl19+0XfffadChQpla/0AANljM66/6BsAAAAAcMc4swUAAAAAFiBsAQCQBVWqVJGbm1umX/Pmzcvt8gAAeRCXEQIAkAWHDh264aPT/fz85O7ufpcrAgDkdYQtAAAAALAAlxECAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABY4P8Ac5PItflxntkAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(x=credit_card_raw['Type_Income'])\n",
        "plt.title(\"Number of people Income Type wise\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "id": "c11SzMehzIFY",
        "outputId": "aec961cd-f8e5-4831-ad1f-eed59a2a9687"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Number of people Income Type wise')"
            ]
          },
          "metadata": {},
          "execution_count": 175
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAHHCAYAAABZbpmkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABWhUlEQVR4nO3deVgVZf8G8Puw7wdR4YAiuCsGmvsRd1EkM33FyiXF3MpAX3el3LLFpVxySzMTMy0z08zMjVwIEYnETFDRUCwF1ARcYhG+vz98mZ9HUAHBg9P9ua5zXc7zPDPznTkLt3Nm5mhEREBERESkUibGLoCIiIioPDHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQERGRqjHsEBERkaox7BAREZGqMewQFdOBAweg0WjwzTffGLuUYklNTUXfvn1RuXJlaDQaLF682NglPZYhQ4bA09PT2GXQU0qj0WDWrFnGLoOMhGGHKpSwsDBoNBpYWVnhr7/+KtTfsWNHPPPMM0ao7Okzbtw47N69G6GhoVi/fj26d+9u7JIqlFmzZkGj0eDq1avGLsVoOnbsCI1G88gHQwI97cyMXQBRUbKzszF37lwsXbrU2KU8tX766Sf06tULEydONHYpVEG99dZbGD58uDIdExODJUuW4M0330TDhg2Vdh8fH2OUV6b++ecfmJnxT96/FZ95qpCaNGmC1atXIzQ0FG5ubsYu54m6desWbG1tH3s5aWlpcHR0fPyCSLW6du1qMG1lZYUlS5aga9eu6Nixo3GKKidWVlbGLoGMiF9jUYX05ptvIi8vD3Pnzn3ouPPnz0Oj0SAsLKxQ3/2H3wu+tjhz5gxeeeUVaLVaVK1aFdOnT4eI4OLFi+jVqxccHByg0+mwYMGCIteZl5eHN998EzqdDra2tnjhhRdw8eLFQuOio6PRvXt3aLVa2NjYoEOHDoiMjDQYU1BTfHw8BgwYgEqVKqFt27YP3eY//vgDL774IpycnGBjY4PWrVvjhx9+UPoLvgoUESxfvlz5KuJR+/DDDz/EokWL4OHhAWtra3To0AG///57ofGnTp1C37594eTkBCsrKzRv3hzbt28vcZ3A/58HtWnTpmLt0/vl5+dj8eLFaNSoEaysrODi4oLXXnsN169ff+S8RSn4mjQ+Ph6dOnWCjY0NqlWrhvnz5xcam5WVhVmzZqFevXqwsrKCq6sr+vTpg3Pnziljbt26hQkTJsDd3R2WlpaoX78+PvzwQ4iIwbI0Gg1CQkKwefNmeHl5wdraGnq9HidOnAAArFq1CnXq1IGVlRU6duyI8+fPF6qnOK+3klq7di00Gg2OHTtWqO/999+Hqamp8nVzwb6LjY1FmzZtYG1tjZo1a2LlypWF5s3OzsbMmTNRp04dWFpawt3dHZMnT0Z2dvZD61myZAlMTU2Rnp6utC1YsAAajQbjx49X2vLy8mBvb48pU6Yobfd/Hty4cQNjx46Fp6cnLC0t4ezsjK5du+LXX381WGd57FcyAiGqQNauXSsAJCYmRoYOHSpWVlby119/Kf0dOnSQRo0aKdNJSUkCQNauXVtoWQBk5syZyvTMmTMFgDRp0kT69+8vK1askB49eggAWbhwodSvX19GjRolK1asEF9fXwEgBw8eVObfv3+/ABBvb2/x8fGRhQsXytSpU8XKykrq1asnt2/fVsaGh4eLhYWF6PV6WbBggSxatEh8fHzEwsJCoqOjC9Xk5eUlvXr1khUrVsjy5csfuH9SUlLExcVF7O3t5a233pKFCxdK48aNxcTERL799lsRETl37pysX79eAEjXrl1l/fr1sn79+gcus2Afent7i6enp8ybN0/efvttcXJykqpVq0pKSooy9vfffxetViteXl4yb948WbZsmbRv3140Go2y/uLWWdJ9GhQUJB4eHga1Dx8+XMzMzGTEiBGycuVKmTJlitja2kqLFi0kJyfngdt8776/cuWK0tahQwdxc3MTd3d3+e9//ysrVqyQzp07CwDZuXOnMu7OnTvSpUsXASD9+vWTZcuWyZw5c6Rz586ybds2ERHJz8+Xzp07i0ajkeHDh8uyZcukZ8+eAkDGjh1rUAsA8fHxEXd3d5k7d67MnTtXtFqt1KhRQ5YtWyZeXl6yYMECmTZtmlhYWEinTp0M5i/u6+1RNm/eLABk//79IiKSmZkp1tbWMmHChEJjvby8pHPnzoX2nbOzs4SEhMiSJUukbdu2AkDWrFmjjMvLy5Nu3bqJjY2NjB07VlatWiUhISFiZmYmvXr1emh9v/76qwCQ77//Xmnr1auXmJiYSPPmzZW2mJgYASA7duxQ2u7/PBgwYIBYWFjI+PHj5dNPP5V58+ZJz5495YsvvlDGlNV+JeNj2KEK5d6wc+7cOTEzM5MxY8Yo/WURdkaOHKm03blzR6pXry4ajUbmzp2rtF+/fl2sra0lKChIaSv4w1ytWjXJzMxU2r/++msBIB999JGI3P0jV7duXfH395f8/Hxl3O3bt6VmzZrStWvXQjX179+/WPtn7NixAkAiIiKUths3bkjNmjXF09NT8vLyDLY/ODj4kcss2IfW1tby559/Ku3R0dECQMaNG6e0denSRby9vSUrK0tpy8/PlzZt2kjdunVLXGdx96lI4bATEREhAGTDhg0G27Nr164i2+/3oLADQD7//HOlLTs7W3Q6nQQGBiptn332mRKS71fwnG/btk0AyLvvvmvQ37dvX9FoNHL27FmlDYBYWlpKUlKS0rZq1SoBIDqdzmDfhIaGCgBlbEleb49yf9gREenfv7+4ubkZvLYKQse977uCfbdgwQKlLTs7W5o0aSLOzs5K+Fy/fr2YmJgYvDZERFauXCkAJDIy8oH15eXliYODg0yePFnZ9sqVK8uLL74opqamcuPGDRERWbhwoZiYmMj169eVee//PNBqtQ99f5TlfiXj49dYVGHVqlULgwYNwieffILLly+X2XLvPSHT1NQUzZs3h4hg2LBhSrujoyPq16+PP/74o9D8gwcPhr29vTLdt29fuLq6YufOnQCAuLg4JCYmYsCAAbh27RquXr2Kq1ev4tatW+jSpQsOHTqE/Px8g2W+/vrrxap9586daNmypcFXXXZ2dhg5ciTOnz+P+Pj44u2EIvTu3RvVqlVTplu2bIlWrVop2/X333/jp59+wksvvYQbN24o23Xt2jX4+/sjMTFR+UqjpHU+ap8WZfPmzdBqtejatatSy9WrV9GsWTPY2dlh//79pdoPdnZ2eOWVV5RpCwsLtGzZ0uC1sGXLFlSpUgWjR48uNH/BV4Y7d+6EqakpxowZY9A/YcIEiAh+/PFHg/YuXboYXFrfqlUrAEBgYKDBviloL6inNK+3khg8eDAuXbpksD83bNgAa2trBAYGGow1MzPDa6+9pkxbWFjgtddeQ1paGmJjYwHcfd4aNmyIBg0aGDxvnTt3BoCHPm8mJiZo06YNDh06BABISEjAtWvXMHXqVIgIoqKiAAARERF45plnHnrOmqOjI6Kjo3Hp0qUi+8t7v9KTxbBDFdq0adNw586dR567UxI1atQwmNZqtbCyskKVKlUKtRd17kfdunUNpjUaDerUqaOcR5GYmAgACAoKQtWqVQ0en376KbKzs5GRkWGwjJo1axar9gsXLqB+/fqF2guunLlw4UKxllOU+7cLAOrVq6ds19mzZyEimD59eqHtmjlzJoC7J0WXps5H7dOiJCYmIiMjA87OzoXquXnzplJLSVWvXr3QOU6VKlUyeC2cO3cO9evXf+jVPRcuXICbm5tBUAEevA+Kel0CgLu7e5HtBfWU5vVWEl27doWrqys2bNgA4O55Ul9++SV69epVaNvc3NwKnVxfr149ADB4f5w8ebJQrQXjHvW8tWvXDrGxsfjnn38QEREBV1dXNG3aFI0bN0ZERAQA4Oeff0a7du0eupz58+fj999/h7u7O1q2bIlZs2YZBNry3q/0ZPFqLKrQatWqhVdeeQWffPIJpk6dWqj/QSfe5uXlPXCZpqamxWoDUOhE0uIo+N/eBx98gCZNmhQ5xs7OzmDa2tq6xOt50gq2a+LEifD39y9yTJ06dZ5oPc7Ozsof4ftVrVq1VMsty9dCWaz3UfWU5vVW0roGDBiA1atXY8WKFYiMjMSlS5cMjn6VRH5+Pry9vbFw4cIi++8Pd/dr27YtcnNzERUVhYiICCXUtGvXDhERETh16hSuXLnyyLDz0ksvoV27dti6dSv27NmDDz74APPmzcO3336LgICAct+v9GQx7FCFN23aNHzxxReYN29eob5KlSoBgMHVGcDjHeF4lIL/8RUQEZw9e1a5F0nt2rUBAA4ODvDz8yvTdXt4eOD06dOF2k+dOqX0l9b92wUAZ86cUb5aqVWrFgDA3Nz8kdtV0joftU+LUrt2bezbtw++vr5PPCzWrl0b0dHRyM3Nhbm5eZFjPDw8sG/fPty4ccPgCEhZPFf31wKUz+utwODBg7FgwQJ8//33+PHHH1G1atUiA++lS5cK3TrhzJkzAKC8jmrXro3jx4+jS5cuD71K8EFatmwJCwsLREREICIiApMmTQIAtG/fHqtXr0Z4eLgy/Siurq5444038MYbbyAtLQ1NmzbFe++9h4CAgCeyX+nJ4ddYVOHVrl0br7zyClatWoWUlBSDPgcHB1SpUkX5Dr/AihUryq2ezz//HDdu3FCmv/nmG1y+fBkBAQEAgGbNmqF27dr48MMPcfPmzULzX7lypdTrfu6553D06FHl3ATg7uXNn3zyCTw9PeHl5VXqZW/bts3grtVHjx5FdHS0sl3Ozs7o2LEjVq1aVeQ5VPduV0nrfNQ+LcpLL72EvLw8vPPOO4X67ty5UygAl6XAwEBcvXoVy5YtK9RXcMTlueeeQ15eXqExixYtgkajeei2lUR5vt4K+Pj4wMfHB59++im2bNmCfv36FfkV3p07d7Bq1SplOicnB6tWrULVqlXRrFkzAHeft7/++gurV68uNP8///yDW7duPbQWKysrtGjRAl9++SWSk5MNjuz8888/WLJkCWrXrg1XV9cHLiMvL6/QV1DOzs5wc3NTLn9/EvuVnhwe2aGnwltvvYX169fj9OnTaNSokUHf8OHDMXfuXAwfPhzNmzfHoUOHlP9NlgcnJye0bdsWr776KlJTU7F48WLUqVMHI0aMAHD3JMpPP/0UAQEBaNSoEV599VVUq1YNf/31F/bv3w8HBwd8//33pVr31KlT8eWXXyIgIABjxoyBk5MT1q1bh6SkJGzZsgUmJqX//0udOnXQtm1bjBo1CtnZ2Vi8eDEqV66MyZMnK2OWL1+Otm3bwtvbGyNGjECtWrWQmpqKqKgo/Pnnnzh+/Hip6nzUPi1Khw4d8Nprr2HOnDmIi4tDt27dYG5ujsTERGzevBkfffQR+vbtW+r98TCDBw/G559/jvHjx+Po0aNo164dbt26hX379uGNN95Ar1690LNnT3Tq1AlvvfUWzp8/j8aNG2PPnj347rvvMHbsWOXIweMqz9fbvQYPHqzcjftBX2G5ublh3rx5OH/+POrVq4dNmzYhLi4On3zyiXIEbNCgQfj666/x+uuvY//+/fD19UVeXh5OnTqFr7/+Grt370bz5s0fWku7du0wd+5caLVaeHt7A7gbVurXr4/Tp09jyJAhD53/xo0bqF69Ovr27YvGjRvDzs4O+/btQ0xMjHJ/rSe1X+kJMdZlYERFuffS8/sFBQUJAINLz0XuXgo6bNgw0Wq1Ym9vLy+99JKkpaU98NLzey81Lliura1tofXdf5l7wWXSX375pYSGhoqzs7NYW1tLjx495MKFC4XmP3bsmPTp00cqV64slpaW4uHhIS+99JKEh4c/sqaHOXfunPTt21ccHR3FyspKWrZsaXA/kQIo4aXnH3zwgSxYsEDc3d3F0tJS2rVrJ8ePHy9y/YMHDxadTifm5uZSrVo1ef755+Wbb74pcZ0l2adF3WdHROSTTz6RZs2aibW1tdjb24u3t7dMnjxZLl269NDtftCl5/e/vh607tu3b8tbb70lNWvWFHNzc9HpdNK3b185d+6cMubGjRsybtw4cXNzE3Nzc6lbt6588MEHBpcyixT9XN37vBS1zzZv3mzQXpzX26MUdel5gcuXL4upqanUq1evyHkL9t0vv/wier1erKysxMPDQ5YtW1ZobE5OjsybN08aNWoklpaWUqlSJWnWrJm8/fbbkpGR8cg6f/jhBwEgAQEBBu3Dhw8vdF+fAvd+HmRnZ8ukSZOkcePGYm9vL7a2ttK4cWNZsWJFofnKYr+S8WlEyvmsOyKq0M6fP4+aNWvigw8+eOK/o3XgwAF06tQJmzdvLrejMFQ2rl69CldXV8yYMQPTp08v1N+xY0dcvXq1yLtuExkbz9khIqJHCgsLQ15eHgYNGmTsUohKjOfsEBHRA/3000+Ij4/He++9h969exvc+JDoacGwQ0REDzR79mwcPnwYvr6+WLp0qbHLISoVnrNDREREqsZzdoiIiEjVGHaIiIhI1XjODu7+VsulS5dgb29fqtuXExER0ZMnIrhx4wbc3NweelNVhh3c/T2XR/34HBEREVVMFy9eRPXq1R/Yz7ADKD/Sd/HiRTg4OBi5GiIiIiqOzMxMuLu7G/zYblEYdgDlqysHBweGHSIioqfMo05B4QnKREREpGoMO0RERKRqDDtERESkagw7REREpGoMO0RERKRqDDtERESkagw7REREpGoMO0RERKRqDDtERESkagw7REREpGpGDTt5eXmYPn06atasCWtra9SuXRvvvPMOREQZIyKYMWMGXF1dYW1tDT8/PyQmJhos5++//8bAgQPh4OAAR0dHDBs2DDdv3nzSm0NEREQVkFHDzrx58/Dxxx9j2bJlSEhIwLx58zB//nwsXbpUGTN//nwsWbIEK1euRHR0NGxtbeHv74+srCxlzMCBA3Hy5Ens3bsXO3bswKFDhzBy5EhjbBIRERFVMBq59zDKE/b888/DxcUFa9asUdoCAwNhbW2NL774AiICNzc3TJgwARMnTgQAZGRkwMXFBWFhYejXrx8SEhLg5eWFmJgYNG/eHACwa9cuPPfcc/jzzz/h5ub2yDoyMzOh1WqRkZHBHwIlIiJ6ShT377dRj+y0adMG4eHhOHPmDADg+PHj+PnnnxEQEAAASEpKQkpKCvz8/JR5tFotWrVqhaioKABAVFQUHB0dlaADAH5+fjAxMUF0dHSR683OzkZmZqbBg4iIiNTJzJgrnzp1KjIzM9GgQQOYmpoiLy8P7733HgYOHAgASElJAQC4uLgYzOfi4qL0paSkwNnZ2aDfzMwMTk5Oypj7zZkzB2+//XZZbw4RERFVQEYNO19//TU2bNiAjRs3olGjRoiLi8PYsWPh5uaGoKCgcltvaGgoxo8fr0xnZmbC3d293NZHRE8X36W+xi6B/idydKSxSyAVMGrYmTRpEqZOnYp+/foBALy9vXHhwgXMmTMHQUFB0Ol0AIDU1FS4uroq86WmpqJJkyYAAJ1Oh7S0NIPl3rlzB3///bcy//0sLS1haWlZDltEREREFY1Rz9m5ffs2TEwMSzA1NUV+fj4AoGbNmtDpdAgPD1f6MzMzER0dDb1eDwDQ6/VIT09HbGysMuann35Cfn4+WrVq9QS2goiIiCoyox7Z6dmzJ9577z3UqFEDjRo1wrFjx7Bw4UIMHToUAKDRaDB27Fi8++67qFu3LmrWrInp06fDzc0NvXv3BgA0bNgQ3bt3x4gRI7By5Urk5uYiJCQE/fr1K9aVWERERKRuRg07S5cuxfTp0/HGG28gLS0Nbm5ueO211zBjxgxlzOTJk3Hr1i2MHDkS6enpaNu2LXbt2gUrKytlzIYNGxASEoIuXbrAxMQEgYGBWLJkiTE2iYiIiCoYo95np6LgfXaI6F48Qbni4AnK9DBPxX12iIiIiMobww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqZpRw46npyc0Gk2hR3BwMAAgKysLwcHBqFy5Muzs7BAYGIjU1FSDZSQnJ6NHjx6wsbGBs7MzJk2ahDt37hhjc4iIiKgCMmrYiYmJweXLl5XH3r17AQAvvvgiAGDcuHH4/vvvsXnzZhw8eBCXLl1Cnz59lPnz8vLQo0cP5OTk4PDhw1i3bh3CwsIwY8YMo2wPERERVTwaERFjF1Fg7Nix2LFjBxITE5GZmYmqVati48aN6Nu3LwDg1KlTaNiwIaKiotC6dWv8+OOPeP7553Hp0iW4uLgAAFauXIkpU6bgypUrsLCwKNZ6MzMzodVqkZGRAQcHh3LbPiJ6Ovgu9TV2CfQ/kaMjjV0CVWDF/ftdYc7ZycnJwRdffIGhQ4dCo9EgNjYWubm58PPzU8Y0aNAANWrUQFRUFAAgKioK3t7eStABAH9/f2RmZuLkyZMPXFd2djYyMzMNHkRERKROFSbsbNu2Denp6RgyZAgAICUlBRYWFnB0dDQY5+LigpSUFGXMvUGnoL+g70HmzJkDrVarPNzd3ctuQ4iIiKhCqTBhZ82aNQgICICbm1u5rys0NBQZGRnK4+LFi+W+TiIiIjIOM2MXAAAXLlzAvn378O233yptOp0OOTk5SE9PNzi6k5qaCp1Op4w5evSowbIKrtYqGFMUS0tLWFpaluEWEBERUUVVIY7srF27Fs7OzujRo4fS1qxZM5ibmyM8PFxpO336NJKTk6HX6wEAer0eJ06cQFpamjJm7969cHBwgJeX15PbACIiIqqwjH5kJz8/H2vXrkVQUBDMzP6/HK1Wi2HDhmH8+PFwcnKCg4MDRo8eDb1ej9atWwMAunXrBi8vLwwaNAjz589HSkoKpk2bhuDgYB65ISIiIgAVIOzs27cPycnJGDp0aKG+RYsWwcTEBIGBgcjOzoa/vz9WrFih9JuammLHjh0YNWoU9Ho9bG1tERQUhNmzZz/JTSAiIqIKrELdZ8dYeJ8dIroX77NTcfA+O/QwT919doiIiIjKA8MOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREama0cPOX3/9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r9fD1tYWQUFBmD17trE2iYiIiCoYjYiIsYswtszMTGi1WmRkZPD8HSKC71JfY5dA/xM5OtLYJVAFVty/3xXqnB0iIiKissawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqxrBDREREqsawQ0RERKrGsENERESqVqqw07lzZ6Snpxdqz8zMROfOnR+3JiIiIqIyU6qwc+DAAeTk5BRqz8rKQkRExGMXRURERFRWzEoy+LffflP+HR8fj5SUFGU6Ly8Pu3btQrVq1cquOiIiIqLHVKKw06RJE2g0Gmg0miK/rrK2tsbSpUvLrDgiIiKix1WisJOUlAQRQa1atXD06FFUrVpV6bOwsICzszNMTU3LvEgiIiKi0ipR2PHw8AAA5Ofnl0sxRERERGWtRGHnXomJidi/fz/S0tIKhZ8ZM2Y8dmFEREREZaFUYWf16tUYNWoUqlSpAp1OB41Go/RpNBqGHSIiIqowShV23n33Xbz33nuYMmVKWddDREREVKZKdZ+d69ev48UXXyzrWoiIiIjKXKnCzosvvog9e/aUdS1EREREZa5UX2PVqVMH06dPx5EjR+Dt7Q1zc3OD/jFjxpRJcURERESPSyMiUtKZatas+eAFajT4448/HquoJy0zMxNarRYZGRlwcHAwdjlEZGS+S32NXQL9T+ToSGOXQBVYcf9+l+rITlJSUqkLIyIiInqSSnXODhEREdHTolRHdoYOHfrQ/s8++6xUxRARERGVtVKFnevXrxtM5+bm4vfff0d6enqRPxBKREREZCylCjtbt24t1Jafn49Ro0ahdu3aj10UERERUVkps3N2TExMMH78eCxatKisFklERET02Mr0BOVz587hzp07ZblIIiIiosdSqq+xxo8fbzAtIrh8+TJ++OEHBAUFlUlhRERERGWhVEd2jh07ZvD47bffAAALFizA4sWLS1XI3LlzodFoMHbsWKUtKysLwcHBqFy5Muzs7BAYGIjU1FSD+ZKTk9GjRw/Y2NjA2dkZkyZN4tElIiIiUpTqyM7+/fvLtIiYmBisWrUKPj4+Bu3jxo3DDz/8gM2bN0Or1SIkJAR9+vRBZOTdO2rm5eWhR48e0Ol0OHz4MC5fvozBgwfD3Nwc77//fpnWSERERE+nxzpn58qVK/j555/x888/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ICoqCo6OjkrQAQA/Pz+YmJggOjr6iddMREREFU+FvRorJSUFAODi4mLQ7uLiovSlpKTA2dnZoN/MzAxOTk7KmKJkZ2cjOztbmc7MzCyrsomIiKiCqbBHdsrTnDlzoNVqlYe7u7uxSyIiIqJyUmHDjk6nAwCkpqYatKempip9Op0OaWlpBv137tzB33//rYwpSmhoKDIyMpTHxYsXy7h6IiIiqigqbNipWbMmdDodwsPDlbbMzExER0dDr9cDAPR6PdLT0xEbG6uM+emnn5Cfn49WrVo9cNmWlpZwcHAweBAREZE6GfWcnZs3b+Ls2bPKdFJSEuLi4uDk5IQaNWpg7NixePfdd1G3bl3UrFkT06dPh5ubG3r37g0AaNiwIbp3744RI0Zg5cqVyM3NRUhICPr168crsYiIiAiAkcPOL7/8gk6dOinT48ePBwAEBQUhLCwMkydPxq1btzBy5Eikp6ejbdu22LVrF6ysrJR5NmzYgJCQEHTp0gUmJiYIDAzEkiVLnvi2EBERUcVk1LDTsWNHiMgD+zUaDWbPno3Zs2c/cIyTkxM2btxYHuURERGRClTYc3aIiIiIygLDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpGsMOERERqRrDDhEREakaww4RERGpmmrCzvLly+Hp6QkrKyu0atUKR48eNXZJREREVAGoIuxs2rQJ48ePx8yZM/Hrr7+icePG8Pf3R1pamrFLIyIiIiNTRdhZuHAhRowYgVdffRVeXl5YuXIlbGxs8Nlnnxm7NCIiIjKypz7s5OTkIDY2Fn5+fkqbiYkJ/Pz8EBUVZcTKiIiIqCIwM3YBj+vq1avIy8uDi4uLQbuLiwtOnTpV5DzZ2dnIzs5WpjMyMgAAmZmZxV5vXvY/paiWykNJnrfSupGVV+7roOJ5Es/3nX/ulPs6qHiexPN96w6f74qipM93wXgReei4pz7s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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(12,4))\n",
        "sns.countplot(x=credit_card_raw['EDUCATION'])\n",
        "plt.title(\"Number of people Education wise\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 427
        },
        "id": "XIIqfsIgN1yD",
        "outputId": "64867f3d-9d09-4c4f-a8dc-01d6a0a5d8e9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Number of people Education wise')"
            ]
          },
          "metadata": {},
          "execution_count": 176
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1200x400 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA/YAAAGJCAYAAAAg86hpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABaFElEQVR4nO3deZhO9f/H8dc9M2YxY2YMZrON3dBoIRp7jEYpKUuKUJYSCVm/WVJESmkRJVvSSqkk2bJkT0ZijEkjxCDLjCXbzOf3h2vOz23GbA0zh+fjuu7rcp/zOee8z/G5l9ec+3yOwxhjBAAAAAAAbMklvwsAAAAAAAC5R7AHAAAAAMDGCPYAAAAAANgYwR4AAAAAABsj2AMAAAAAYGMEewAAAAAAbIxgDwAAAACAjRHsAQAAAACwMYI9AAAAAAA2RrAHANyQVqxYIYfDoblz5+Z3Kdly6NAhtWnTRsWKFZPD4dDEiRPzu6T/pEuXLgoLC8vvMixp/WHFihX5XcpVzZw5Uw6HQ3v27MnvUrLFDscUAG4WBHsAQK6lBRFPT0/9/fff6eY3btxYt9xySz5UZj/9+vXTjz/+qKFDh2r27Nlq3rx5fpdUoLz44otyOBxXfSQmJuZ3idn2yiuvaP78+fldBgDgBuKW3wUAAOzv3LlzGjdunN555538LsW2li9frgcffFADBgzI71IKtMmTJ8vHxyfddH9//+tfTC698soratOmjVq1auU0/fHHH1f79u3l4eGRP4XlUMOGDfXvv//K3d09v0sBgJsewR4A8J/ddtttmjp1qoYOHarQ0ND8Lue6On36tLy9vf/zeg4fPmyrcJpf2rRpo+LFi+d3GdeEq6urXF1d87uMbHNxcZGnp2d+lwEAED/FBwDkgf/9739KSUnRuHHjMm23Z88eORwOzZw5M908h8OhF1980Xqe9tPrXbt2qWPHjvLz81OJEiU0fPhwGWO0b98+Pfjgg/L19VVwcLAmTJiQ4TZTUlL0v//9T8HBwfL29lbLli21b9++dO02bNig5s2by8/PT4ULF1ajRo20Zs0apzZpNe3YsUOPPfaYihYtqvr162e6z3/++afatm2rgIAAFS5cWHfddZe+//57a37a5QzGGE2aNMn6aXlWx/D111/Xm2++qbJly8rLy0uNGjXS77//nq79zp071aZNGwUEBMjT01O1atXSt99+m+M6pf+/pvrzzz/P1jG9UmpqqiZOnKjq1avL09NTQUFBeuqpp3T8+PEsl82J/fv3q1WrVvL29lZgYKD69eunc+fOpWsXFhamLl26pJveuHFjNW7c2Gna2bNn9eKLL6py5cry9PRUSEiIHn74Ye3evdtq8/rrr6tu3boqVqyYvLy8VLNmzXRjPDgcDp0+fVqzZs2y/q/TarjaNfbvvfeeqlevLg8PD4WGhqpXr146ceJEuppvueUW7dixQ3fffbcKFy6skiVLavz48Vker4cfflh33HGH07QHHnhADofDqa9s2LBBDodDP/zwg6SMr7GPj49X69atFRwcLE9PT5UqVUrt27dXUlKS0/o//vhj1axZU15eXgoICFD79u2z1YcAABnjjD0A4D8rV66cOnXqpKlTp2rIkCF5etb+kUceUXh4uMaNG6fvv/9eo0ePVkBAgN5//301adJEr776qubMmaMBAwbozjvvVMOGDZ2WHzNmjBwOhwYPHqzDhw9r4sSJioqKUkxMjLy8vCRd+hn8vffeq5o1a2rkyJFycXHRjBkz1KRJE61evVq1a9d2Wmfbtm1VqVIlvfLKKzLGXLX2Q4cOqW7dujpz5oz69OmjYsWKadasWWrZsqXmzp2rhx56SA0bNtTs2bP1+OOPq1mzZurUqVO2jstHH32kkydPqlevXjp79qzeeustNWnSRNu2bVNQUJAkafv27apXr55KliypIUOGyNvbW1988YVatWqlefPm6aGHHsp2nTk9phl56qmnNHPmTD3xxBPq06ePEhIS9O6772rLli1as2aNChUqlOV+Hzt2LN00Nzc369cO//77r5o2baq9e/eqT58+Cg0N1ezZs7V8+fJsHdeMpKSk6P7779eyZcvUvn17Pffcczp58qSWLFmi33//XRUqVJAkvfXWW2rZsqU6dOig8+fP67PPPlPbtm21YMECtWjRQpI0e/ZsdevWTbVr11aPHj0kyVo+Iy+++KJGjRqlqKgo9ezZU3FxcZo8ebI2bdqU7pgdP35czZs318MPP6x27dpp7ty5Gjx4sCIiInTvvfdedRsNGjTQN998o+TkZPn6+soYozVr1sjFxUWrV69Wy5YtJUmrV6+Wi4uL6tWrl+F6zp8/r+joaJ07d07PPvusgoOD9ffff2vBggU6ceKE/Pz8JF3qP8OHD1e7du3UrVs3HTlyRO+8844aNmyoLVu28MsVAMgNAwBALs2YMcNIMps2bTK7d+82bm5upk+fPtb8Ro0amerVq1vPExISjCQzY8aMdOuSZEaOHGk9HzlypJFkevToYU27ePGiKVWqlHE4HGbcuHHW9OPHjxsvLy/TuXNna9pPP/1kJJmSJUua5ORka/oXX3xhJJm33nrLGGNMamqqqVSpkomOjjapqalWuzNnzphy5cqZZs2apavp0Ucfzdbx6du3r5FkVq9ebU07efKkKVeunAkLCzMpKSlO+9+rV68s15l2DL28vMz+/fut6Rs2bDCSTL9+/axpTZs2NREREebs2bPWtNTUVFO3bl1TqVKlHNeZ3WNqjDGdO3c2ZcuWtZ6vXr3aSDJz5sxx2p9FixZlOP1Kacc+o0eVKlWsdhMnTjSSzBdffGFNO336tKlYsaKRZH766SdretmyZZ36TJpGjRqZRo0aWc+nT59uJJk33ngjXdsr+8zlzp8/b2655RbTpEkTp+ne3t4Zbjft9ZSQkGCMMebw4cPG3d3d3HPPPU595d133zWSzPTp051qlmQ++ugja9q5c+dMcHCwad26dbptXW7Tpk1Gklm4cKExxpjffvvNSDJt27Y1derUsdq1bNnS3H777dbztP6Qdky3bNliJJkvv/zyqtvas2ePcXV1NWPGjHGavm3bNuPm5pZuOgAge/gpPgAgT5QvX16PP/64PvjgAx08eDDP1tutWzfr366urqpVq5aMMeratas13d/fX1WqVNGff/6ZbvlOnTqpSJEi1vM2bdooJCRECxculCTFxMQoPj5ejz32mI4ePap//vlH//zzj06fPq2mTZtq1apVSk1NdVrn008/na3aFy5cqNq1azv9XN/Hx0c9evTQnj17tGPHjuwdhAy0atVKJUuWtJ7Xrl1bderUsfbr2LFjWr58udq1a6eTJ09a+3X06FFFR0crPj7eupNBTuvM6phm5Msvv5Sfn5+aNWtm1fLPP/+oZs2a8vHx0U8//ZSt/Z43b56WLFni9JgxY4Y1f+HChQoJCVGbNm2saYULF7bOjufGvHnzVLx4cT377LPp5l1+2cTlv1Y4fvy4kpKS1KBBA/3666+52u7SpUt1/vx59e3bVy4u//+VrXv37vL19U13qYSPj486duxoPXd3d1ft2rUzfF1c7vbbb5ePj49WrVol6dKZ+VKlSqlTp0769ddfdebMGRlj9PPPP6tBgwZXXU/aGfkff/xRZ86cybDNV199pdTUVLVr186pHwQHB6tSpUrZ7gcAAGf8FB8AkGeGDRum2bNna9y4cXrrrbfyZJ1lypRxeu7n5ydPT890A6j5+fnp6NGj6ZavVKmS03OHw6GKFSta1zHHx8dLkjp37nzVGpKSklS0aFHrebly5bJV+19//aU6deqkmx4eHm7Nz+3tAK/cL0mqXLmyvvjiC0nSH3/8IWOMhg8fruHDh2e4jsOHD6tkyZI5rjOrY5qR+Ph4JSUlKTAw8Kq1ZEfDhg0zHTzvr7/+UsWKFdONU1ClSpVsrT8ju3fvVpUqVeTmlvnXpgULFmj06NGKiYlxuqY/szETMvPXX39JSl+7u7u7ypcvb81PU6pUqXTbKlq0qH777bdMt+Pq6qrIyEitXr1a0qVg36BBA9WvX18pKSlav369goKCdOzYsUyDfbly5dS/f3+98cYbmjNnjho0aKCWLVtaY2RIl/qBMSbD/ispW5djAADSI9gDAPJM+fLl1bFjR33wwQcaMmRIuvlXCzgpKSlXXWdGo4RfbeRwk8n17leTdjb+tdde02233ZZhmytvr5bZdeQFRdp+DRgwQNHR0Rm2qVix4nWtJzAwUHPmzMlwfokSJa5bLWky6485HZ0+7Vr0hg0b6r333lNISIgKFSqkGTNm6JNPPsmLcrP0X14X9evX15gxY3T27FmtXr1aL7zwgvz9/XXLLbdo9erV1rgNmQV7SZowYYK6dOmib775RosXL1afPn00duxYrV+/XqVKlVJqaqo1AF9G9WZ0K0MAQNYI9gCAPDVs2DB9/PHHevXVV9PNSzvrfeWI3leeecxLaWfk0xhj9Mcff6hGjRqS/n/gMl9fX0VFReXptsuWLau4uLh003fu3GnNz60r90uSdu3apbCwMEmX/sgiXToDmtV+5bTOrI5pRipUqKClS5eqXr161/QPI2XLltXvv/8uY4xTcM9o/4oWLZquL0qX+mPa8ZMu1b5hwwZduHDhqmeU582bJ09PT/34449O96G//DKBNNk9g5923OPi4pzqOX/+vBISEvK0vzZo0EDnz5/Xp59+qr///tsK8A0bNrSCfeXKla2An5mIiAhFRERo2LBhWrt2rerVq6cpU6Zo9OjRqlChgowxKleunCpXrpxn9QPAzY5r7AEAeapChQrq2LGj3n//fSUmJjrN8/X1VfHixa1redO8995716yetNHj08ydO1cHDx60RgmvWbOmKlSooNdff12nTp1Kt/yRI0dyve377rtPGzdu1Lp166xpp0+f1gcffKCwsDBVq1Yt1+ueP3++dY28JG3cuFEbNmyw9iswMFCNGzfW+++/n+GYB5fvV07rzOqYZqRdu3ZKSUnRyy+/nG7exYsXMwzYuXHffffpwIEDTreZO3PmjD744IN0bStUqKD169fr/Pnz1rQFCxaku+1a69at9c8//+jdd99Nt460s+Gurq5yOBxOvz7Zs2eP5s+fn24Zb2/vbO1vVFSU3N3d9fbbbzuddZ82bZqSkpKskfbzQp06dVSoUCG9+uqrCggIUPXq1SVdCvzr16/XypUrszxbn5ycrIsXLzpNi4iIkIuLi3VpwsMPPyxXV1eNGjUq3S8JjDEZXk4DAMgaZ+wBAHnuhRde0OzZsxUXF2cFhDTdunXTuHHj1K1bN9WqVUurVq3Srl27rlktAQEBql+/vp544gkdOnRIEydOVMWKFdW9e3dJkouLiz788EPde++9ql69up544gmVLFlSf//9t3766Sf5+vrqu+++y9W2hwwZok8//VT33nuv+vTpo4CAAM2aNUsJCQmaN2+e04BoOVWxYkXVr19fPXv21Llz5zRx4kQVK1ZMgwYNstpMmjRJ9evXV0REhLp3767y5cvr0KFDWrdunfbv36+tW7fmqs6sjmlGGjVqpKeeekpjx45VTEyM7rnnHhUqVEjx8fH68ssv9dZbbzkNeHc1c+fOzfDn2s2aNVNQUJC6d++ud999V506ddLmzZsVEhKi2bNnq3DhwumW6datm+bOnavmzZurXbt22r17tz7++ON0t5/r1KmTPvroI/Xv318bN25UgwYNdPr0aS1dulTPPPOMHnzwQbVo0UJvvPGGmjdvrscee0yHDx/WpEmTVLFixXTXuNesWVNLly7VG2+8odDQUJUrVy7DMQ5KlCihoUOHatSoUWrevLlatmypuLg4vffee7rzzjudBsr7rwoXLqyaNWtq/fr11j3spUtn7E+fPq3Tp09nGeyXL1+u3r17q23btqpcubIuXryo2bNny9XVVa1bt5Z06Y8po0eP1tChQ7Vnzx61atVKRYoUUUJCgr7++mv16NFDAwYMyLP9AoCbRn4MxQ8AuDFcfru7K3Xu3NlIcrrdnTGXbgnWtWtX4+fnZ4oUKWLatWtnDh8+fNXb3R05ciTder29vdNt78pb66XdiuvTTz81Q4cONYGBgcbLy8u0aNHC/PXXX+mW37Jli3n44YdNsWLFjIeHhylbtqxp166dWbZsWZY1ZWb37t2mTZs2xt/f33h6epratWubBQsWpGunHN7u7rXXXjMTJkwwpUuXNh4eHqZBgwZm69atGW6/U6dOJjg42BQqVMiULFnS3H///Wbu3Lk5rjMnx/TK292l+eCDD0zNmjWNl5eXKVKkiImIiDCDBg0yBw4cyHS/M7vdna64jd1ff/1lWrZsaQoXLmyKFy9unnvuOeu2epe3M8aYCRMmmJIlSxoPDw9Tr14988svv6S73Z0xl/rtCy+8YMqVK2cKFSpkgoODTZs2bczu3butNtOmTTOVKlUyHh4epmrVqmbGjBlW3ZfbuXOnadiwofHy8jKSrFvfXXm7uzTvvvuuqVq1qilUqJAJCgoyPXv2NMePH3dqc2X/T3O1/4eMDBw40Egyr776qtP0tFsFXr6vxqS/3d2ff/5pnnzySVOhQgXj6elpAgICzN13322WLl2ablvz5s0z9evXN97e3sbb29tUrVrV9OrVy8TFxWWrVgCAM4cxuRhpCAAA5Is9e/aoXLlyeu211677mc0VK1bo7rvv1pdffpmts+sAAOD64Bp7AAAAAABsjGAPAAAAAICNEewBAAAAALAxrrEHAAAAAMDGOGMPAAAAAICNEewBAAAAALAxt/wuwA5SU1N14MABFSlSRA6HI7/LAQAAAADc4IwxOnnypEJDQ+Xikvk5eYJ9Nhw4cEClS5fO7zIAAAAAADeZffv2qVSpUpm2IdhnQ5EiRSRdOqC+vr75XA0AAAAA4EaXnJys0qVLW3k0MwT7bEj7+b2vry/BHgAAAABw3WTncnAGzwMAAAAAwMYI9gAAAAAA2BjBHgAAAAAAGyPYAwAAAABgYwR7AAAAAABsjGAPAAAAAICNEewBAAAAALAxgj0AAAAAADZGsAcAAAAAwMYI9gAAAAAA2BjBHgAAAAAAG3PL7wIA3Jj2vhSR3yXgBlNmxLb8LgEAAKBA4ow9AAAAAAA2RrAHAAAAAMDGCPYAAAAAANgYwR4AAAAAABvL12C/atUqPfDAAwoNDZXD4dD8+fOd5htjNGLECIWEhMjLy0tRUVGKj493anPs2DF16NBBvr6+8vf3V9euXXXq1CmnNr/99psaNGggT09PlS5dWuPHj7/WuwYAAAAAwHWRr8H+9OnTuvXWWzVp0qQM548fP15vv/22pkyZog0bNsjb21vR0dE6e/as1aZDhw7avn27lixZogULFmjVqlXq0aOHNT85OVn33HOPypYtq82bN+u1117Tiy++qA8++OCa7x8AAAAAANeawxhj8rsISXI4HPr666/VqlUrSZfO1oeGhur555/XgAEDJElJSUkKCgrSzJkz1b59e8XGxqpatWratGmTatWqJUlatGiR7rvvPu3fv1+hoaGaPHmyXnjhBSUmJsrd3V2SNGTIEM2fP187d+7MVm3Jycny8/NTUlKSfH19837ngRsQt7tDXuN2dwAA4GaSkxxaYK+xT0hIUGJioqKioqxpfn5+qlOnjtatWydJWrdunfz9/a1QL0lRUVFycXHRhg0brDYNGza0Qr0kRUdHKy4uTsePH89w2+fOnVNycrLTAwAAAACAgqjABvvExERJUlBQkNP0oKAga15iYqICAwOd5ru5uSkgIMCpTUbruHwbVxo7dqz8/PysR+nSpf/7DgEAAAAAcA0U2GCfn4YOHaqkpCTrsW/fvvwuCQAAAACADBXYYB8cHCxJOnTokNP0Q4cOWfOCg4N1+PBhp/kXL17UsWPHnNpktI7Lt3ElDw8P+fr6Oj0AAAAAACiICmywL1eunIKDg7Vs2TJrWnJysjZs2KDIyEhJUmRkpE6cOKHNmzdbbZYvX67U1FTVqVPHarNq1SpduHDBarNkyRJVqVJFRYsWvU57AwAAAADAtZGvwf7UqVOKiYlRTEyMpEsD5sXExGjv3r1yOBzq27evRo8erW+//Vbbtm1Tp06dFBoaao2cHx4erubNm6t79+7auHGj1qxZo969e6t9+/YKDQ2VJD322GNyd3dX165dtX37dn3++ed666231L9//3zaawAAAAAA8o5bfm78l19+0d133209TwvbnTt31syZMzVo0CCdPn1aPXr00IkTJ1S/fn0tWrRInp6e1jJz5sxR79691bRpU7m4uKh169Z6++23rfl+fn5avHixevXqpZo1a6p48eIaMWKE073uAQAAAACwqwJzH/uCjPvYAznHfeyR17iPPQAAuJncEPexBwAAAAAAWSPYAwAA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ZXsHDx7Uvffee122haytW7dOrq6uatGiRX6Xkqfs8J0XN9Ht7m5GXl5e8vLyyu8ylJKSIofDIReXG+vvSAXl+GamdevWOn/+vGbNmqXy5cvr0KFDWrZsmY4ePZrfpV0XFy5cUKFChXK1bEpKihYsWKDvv/8+j6uyt/Pnz8vd3T2/y8gxd3d3BQcH53cZtmPX/+9rheOBgqagvK/9l8/bG8m0adP07LPPatq0aTpw4IBCQ0Pzu6Q8YYfvvBkxxiglJUVubjdH5L2xkhacZPSzmdGjRyswMFBFihRRt27dNGTIkAx/BvP6668rJCRExYoVU69evXThwgVr3rlz5zRgwACVLFlS3t7eqlOnjlasWJFuu99++62qVasmDw8P7d27N8Maf//9d917773y8fFRUFCQHn/8cf3zzz/W/NOnT6tTp07y8fFRSEiIJkyYkG4dGf0Mzd/fXzNnzrSe79+/X48++qgCAgLk7e2tWrVqacOGDZKk3bt368EHH1RQUJB8fHx05513aunSpdayjRs31l9//aV+/frJ4XDI4XBc9fhOnjxZFSpUkLu7u6pUqaLZs2enq/XDDz/UQw89pMKFC6tSpUr69ttvMzw2/9WJEye0evVqvfrqq7r77rtVtmxZ1a5dW0OHDlXLli2d2nXr1k0lSpSQr6+vmjRpoq1btzqt67vvvtOdd94pT09PFS9eXA899JA17/jx4+rUqZOKFi2qwoUL695771V8fLw1P+04/fjjjwoPD5ePj4+aN2+ugwcPWm1SUlLUv39/+fv7q1ixYho0aJCuvBPnokWLVL9+favN/fffr927d1vz086SfP7552rUqJE8PT31wQcfyNfXV3PnznVa1/z58+Xt7a2TJ09e9fitXbtWhQoV0p133pnh/Llz5yoiIkJeXl4qVqyYoqKidPr0aWv+hx9+qPDwcHl6eqpq1ap67733nJbPrE9KedOXFi5cqMqVK8vLy0t333239uzZ4zT/6NGjevTRR1WyZEkVLlxYERER+vTTT53aNG7cWL1791bfvn1VvHhxRUdH68knn9T999/v1O7ChQsKDAzUtGnTMjxef/31lx544AEVLVpU3t7eql69uhYuXCjp/39t9P3336tGjRry9PTUXXfdpd9//91pHT///LMaNGggLy8vlS5dWn369HE65ufOndPgwYNVunRpeXh4qGLFilY9V/6iKTv7XhCsXLlStWvXloeHh0JCQjRkyBBdvHhRkrRgwQL5+/srJSVFkhQTEyOHw6EhQ4ZYy3fr1k0dO3a0nmd1DMPCwvTyyy+rU6dO8vX1VY8ePTKsi/6fs/5/s2vcuLH69OmjQYMGKSAgQMHBwXrxxRed2pw4cUJPPfWUgoKC5OnpqVtuuUULFiyw5s+bN0/Vq1eXh4eHwsLC0n0fCAsL0+jRo63vDGXLltW3336rI0eO6MEHH5SPj49q1KihX375xVom7fNp/vz5qlSpkjw9PRUdHa19+/Zluj+Z9e9y5cpJkm6//XY5HA41btw4W8tdTWpqaqbH7crvQGvXrtVtt90mT09P1apVS/Pnz8/wFwSbN29WrVq1VLhwYdWtW1dxcXFO87/55hvdcccd8vT0VPny5TVq1CjrvSdtu5MnT1bLli3l7e2tMWPGZLkvN7pTp07p888/V8+ePdWiRQun76FpMvs+NXv2bNWqVUtFihRRcHCwHnvsMR0+fNhp+aze16Tsvc/n9rWS3X3JyLhx4xQUFKQiRYqoa9euOnv2bLo2Wb1GsurfaZ/1P/zwg2rWrCkPDw/9/PPPSk1N1dixY1WuXDl5eXnp1ltvTffdMKtMYgsGttO5c2fz4IMPppv+008/GUnm+PHjxhhjZsyYYfz8/Kz5H3/8sfH09DTTp083cXFxZtSoUcbX19fceuutTuv29fU1Tz/9tImNjTXfffedKVy4sPnggw+sNt26dTN169Y1q1atMn/88Yd57bXXjIeHh9m1a5e13UKFCpm6deuaNWvWmJ07d5rTp0+nq/f48eOmRIkSZujQoSY2Ntb8+uuvplmzZubuu++22vTs2dOUKVPGLF261Pz222/m/vvvN0WKFDHPPfec1UaS+frrr53W7efnZ2bMmGGMMebkyZOmfPnypkGDBmb16tUmPj7efP7552bt2rXGGGNiYmLMlClTzLZt28yuXbvMsGHDjKenp/nrr7+MMcYcPXrUlCpVyrz00kvm4MGD5uDBgxke36+++soUKlTITJo0ycTFxZkJEyYYV1dXs3z5cqdaS5UqZT755BMTHx9v+vTpY3x8fMzRo0fTHZ//6sKFC8bHx8f07dvXnD179qrtoqKizAMPPGA2bdpkdu3aZZ5//nlTrFgxq6YFCxYYV1dXM2LECLNjxw4TExNjXnnlFWv5li1bmvDwcLNq1SoTExNjoqOjTcWKFc358+eNMf/fH6KiosymTZvM5s2bTXh4uHnsscesdbz66qumaNGiZt68eWbHjh2ma9eupkiRIk79fO7cuWbevHkmPj7ebNmyxTzwwAMmIiLCpKSkGGOMSUhIMJJMWFiYmTdvnvnzzz/NgQMHTPfu3c19993ntM8tW7Y0nTp1yvT4DRgwwPTo0SPDeQcOHDBubm7mjTfeMAkJCea3334zkyZNMidPnjTGXHqthYSEWHXMmzfPBAQEmJkzZxpjsu6TedGX9u7dazw8PEz//v3Nzp07zccff2yCgoKc3iP2799vXnvtNbNlyxaze/du8/bbbxtXV1ezYcMGazuNGjUyPj4+ZuDAgWbnzp1m586dZs2aNcbV1dUcOHDAavfVV18Zb29v6xhcqUWLFqZZs2bmt99+M7t37zbfffedWblypTHm/9+7wsPDzeLFi63XelhYmNWP/vjjD+Pt7W3efPNNs2vXLrNmzRpz++23my5duljbaNeunSldurT56quvzO7du83SpUvNZ5995rSNnO775e8118LV3s/TaixcuLB55plnTGxsrPn6669N8eLFzciRI40xxpw4ccK4uLiYTZs2GWOMmThxoilevLipU6eOtY6KFSuaqVOnGmOydwzLli1rfH19zeuvv27++OMP88cff6Sri/6f8/5/s7myXzdq1Mj4+vqaF1980ezatcvMmjXLOBwOs3jxYmOMMSkpKeauu+4y1atXN4sXL7beIxYuXGiMMeaXX34xLi4u5qWXXjJxcXFmxowZxsvLy/qcN+ZS3w0ICDBTpkwxu3btMj179jS+vr6mefPm5osvvjBxcXGmVatWJjw83KSmphpj/v/zqVatWmbt2rXml19+MbVr1zZ169a11jty5Ein70lZ9e+NGzcaSWbp0qXm4MGDVp/MarmMZHXcjHH+DpSUlGQCAgJMx44dzfbt283ChQtN5cqVjSSzZcsWY8z/vxfWqVPHrFixwmzfvt00aNDAaZ9XrVplfH19zcyZM83u3bvN4sWLTVhYmHnxxRedthsYGGimT59udu/ebX1nuplNmzbN1KpVyxhjzHfffWcqVKhg9TVjsv4+NW3aNLNw4UKze/dus27dOhMZGWnuvfdea3523tey+z6fm9fK5d95s9qXK33++efGw8PDfPjhh2bnzp3m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6dJrtHPnznryySc1f/58JSQkaMWKFfriiy8yrOe/7nteSkpKSndp1b59+/TMM89o3759evbZZ7Vz50598803GjlypPr372+9DxYtWlQ1atTQnDlzrBDfsGFD/frrr9q1a5fTGfusjmF20f8vyUn/R+YaNWqkhg0bqnXr1lqyZIkSEhL0ww8/aNGiRZKk559/XsuWLdPLL7+sXbt2adasWXr33Xfz5PO1UKFCevbZZ7VhwwZt3rxZXbp00V133XXVn/hm1b8DAwPl5eWlRYsW6dChQ0pKSsrWcnkh7ftLjx49FBsbqx9//FGvv/66JOXou96IESP00UcfadSoUdq+fbtiY2P12WefadiwYXlW641kwYIFOn78uLp27apbbrnF6dG6dWvr5/iZfZ8qU6aM3N3d9c477+jPP//Ut99+q5dfftlpO9l5X8ur9/msZLYvGXnuuec0ffp0zZgxQ7t27dLIkSO1fft2pzZZvUZy27+LFCmiAQMGqF+/fpo1a5Z2796tX3/9Ve+88451mWZWmcQ28vUKf+RKbkfFN8aYl156yRQvXtz4+PiYJ5980vTp08fcddddma77ueeeM40aNbKenz9/3owYMcKEhYWZQoUKmZCQEPPQQw+Z33777arbvZpdu3aZhx56yPj7+xsvLy9TtWpV07dvX2uwqJMnT5qOHTuawoULm6CgIDN+/Ph0A1r9/fff5p577jHe3t6mUqVKZuHChU6D5xljzJ49e0zr1q2Nr6+vKVy4sKlVq5Y1ME5CQoK5++67jZeXlyldurR59913021j3bp1pkaNGsbDw8OkvWwy2s/33nvPlC9f3hQqVMhUrlzZfPTRR07zlcVAf3np7NmzZsiQIeaOO+4wfn5+pnDhwqZKlSpm2LBh5syZM1a75ORk8+yzz5rQ0FBTqFAhU7p0adOhQwezd+9eq828efPMbbfdZtzd3U3x4sXNww8/bM07duyYefzxx42fn5/x8vIy0dHR1mBdxmR8nL7++mtz+dvPhQsXzHPPPWd8fX2Nv7+/6d+/v+nUqZNTX1yyZIkJDw83Hh4epkaNGmbFihVOx/NqAxalWbZsmZHkNBhjRj788ENTr169TNvs2LHDREdHmxIlShgPDw9TuXJlp8EWjTFmzpw51jErWrSoadiwofnqq6+s+Zn1SWPypi999913pmLFisbDw8M0aNDA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9e/YsUPr16/XoEGD5ObmJjc3N9111106c+aMPvvss9zspipXrqzY2NgM56VNr1y5crp5FSpUUPfu3TVkyJBc/WEAAICCimAPAMBN7K233pKLi4tatWolSapUqZJcXFy0efNmp3Z//vmnkpKSrMDcpk0bubu7a/z48RmuN23wvGnTpqlhw4baunWrYmJirEf//v1z/XP89u3ba+nSpemuo09NTdWbb76patWqpbv+Ps2IESO0a9euXP9RAQCAgoif4gMAcIM4d+6cdY/4NG5ubipevLikS4PYJSYm6sKFC0pISNDHH3+sDz/8UGPHjlXFihUlXRo5v1u3bnr++efl5uamiIgI7du3T4MHD9Zdd91l3X6udOnSevPNN9W7d28lJyerU6dOCgsL0/79+/XRRx/Jx8dH48aN0+zZs/XSSy/plltucaqrW7dueuONN7R9+3ZVr149R/vZr18/ffPNN3rggQc0YcIE1alTR4cOHdIrr7yi2NhYLV269KqXCQQFBal///567bXXcrRNAAAKMs7YAwBwg1i0aJFCQkKcHvXr17fmjxgxQiEhIapYsaIef/xxJSUladmyZekGlHvrrbfUuXNnDR48WNWrV1eXLl1Uo0YNfffdd06B+ZlnntHixYv1999/66GHHlLVqlXVrVs3+fr6asCAAfr222919OhRPfTQQ+lqDQ8PV3h4eK7O2nt6emr58uXq1KmT/ve//6lixYpq3ry5XF1dtX79et11112ZLj9gwAD5+PjkeLsAABRUDsNFZgAAAAAA2BZn7AEAAAAAsDGCPQAAAAAANkawBwAAAADAxgj2AAAAAADYGMEeAAAAAAAbI9gDAAAAAGBjBHsAAAAAAGyMYA8AAAAAgI0R7AEAAAAAsDGCPQAAAAAANkawBwAAAADAxv4PZWG9OFT+p8cAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns\n",
        "sns.barplot(data=credit_card_raw,x='GENDER',y='Annual_income',hue='label')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "DU21a4yDPG2c",
        "outputId": "8b0825ef-b6ac-44bd-c387-a701cc995ca3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='GENDER', ylabel='Annual_income'>"
            ]
          },
          "metadata": {},
          "execution_count": 184
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlYAAAGwCAYAAABrUCsdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA230lEQVR4nO3de3QU9f3/8dcmkBuQBAhJSAkhlAoi9yCYKgqSLwumYoS2gLRyE5QmCsSC0tqA0BaKysWCcKyVQL9Qke9PqKIGaLiJBJFAuAkoGhs5ZJPIJYEACSTz+8NmyjYIWZhls+H5OGfOycy89zPv3Z4tL2c+M2szDMMQAAAAbpqPpxsAAACoKwhWAAAAFiFYAQAAWIRgBQAAYBGCFQAAgEUIVgAAABYhWAEAAFiknqcbuJ1UVlbqxIkTatSokWw2m6fbAQAANWAYhs6ePauoqCj5+Fz7nBTB6hY6ceKEoqOjPd0GAAC4Ad98841atGhxzRqC1S3UqFEjSd/9DxMcHOzhbgAAQE2UlJQoOjra/Hf8WghWt1DV5b/g4GCCFQAAXqYm03iYvA4AAGARghUAAIBFCFYAAAAWYY4VAAC4roqKCl26dMnTbbhF/fr15evra8lYBCsAAPC9DMOQw+HQmTNnPN2KW4WGhioyMvKmnzNJsAIAAN+rKlSFh4crKCiozj3g2jAMnT9/XoWFhZKk5s2b39R4BCsAAHBVFRUVZqhq2rSpp9txm8DAQElSYWGhwsPDb+qyIJPXAQDAVVXNqQoKCvJwJ+5X9R5vdh6ZR4PVrFmzdPfdd6tRo0YKDw9XUlKSjh496lTTu3dv2Ww2p+Wpp55yqsnLy1NiYqKCgoIUHh6uyZMn6/Lly041W7ZsUbdu3eTv7682bdooPT29Wj+LFi1Sq1atFBAQoJ49e2rXrl1O+y9evKjk5GQ1bdpUDRs21ODBg1VQUGDNhwEAQC1V1y7/XY1V79GjwWrr1q1KTk7Wzp07tXHjRl26dEn9+vVTaWmpU93YsWOVn59vLnPmzDH3VVRUKDExUeXl5dqxY4eWLVum9PR0paWlmTW5ublKTExUnz59lJOTo4kTJ+qJJ57Q+vXrzZpVq1YpNTVV06ZN0549e9S5c2fZ7XbzmqskTZo0Se+9955Wr16trVu36sSJExo0aJAbPyEAAOBVjFqksLDQkGRs3brV3PbAAw8YEyZM+N7XfPDBB4aPj4/hcDjMbYsXLzaCg4ONsrIywzAMY8qUKcZdd93l9LohQ4YYdrvdXO/Ro4eRnJxsrldUVBhRUVHGrFmzDMMwjDNnzhj169c3Vq9ebdYcPnzYkGRkZWVdtbeLFy8axcXF5vLNN98Ykozi4uIafBoAAHjWhQsXjM8++8y4cOGCp1txu2u91+Li4hr/+12r5lgVFxdLkpo0aeK0fcWKFQoLC1OHDh00depUnT9/3tyXlZWljh07KiIiwtxmt9tVUlKiQ4cOmTUJCQlOY9rtdmVlZUmSysvLlZ2d7VTj4+OjhIQEsyY7O1uXLl1yqmnXrp1atmxp1vy3WbNmKSQkxFyio6Nd/kwAAPBGvXv31sSJE2tUu2XLFtlstpt+pEOrVq00f/78mxrjZtWauwIrKys1ceJE3XvvverQoYO5/bHHHlNMTIyioqK0f/9+Pffcczp69KjeeecdSd/dBnplqJJkrjscjmvWlJSU6MKFCzp9+rQqKiquWnPkyBFzDD8/P4WGhlarqTrOf5s6dapSU1PN9apfxwYAAHVTrQlWycnJOnjwoLZv3+60fdy4cebfHTt2VPPmzdW3b199+eWX+uEPf3ir23SJv7+//P39Pd0GcMMmTJigoqIiSVKzZs20YMECD3cEALVbrbgUmJKSonXr1mnz5s1q0aLFNWt79uwpSTp27JgkKTIystqdeVXrkZGR16wJDg5WYGCgwsLC5Ovre9WaK8coLy+vdpryyhqgrikqKlJBQYEKCgrMgAUArvrb3/6m7t27q1GjRoqMjNRjjz3mdHNYlY8//lidOnVSQECA7rnnHh08eNBp//bt29WrVy8FBgYqOjpazzzzTLUb3jzNo8HKMAylpKRozZo12rRpk2JjY6/7mpycHEn/eTJqfHy8Dhw44PQ/0MaNGxUcHKz27dubNZmZmU7jbNy4UfHx8ZIkPz8/xcXFOdVUVlYqMzPTrImLi1P9+vWdao4ePaq8vDyzBgAAVHfp0iXNnDlT+/bt09q1a/X1119r5MiR1eomT56sV155RZ9++qmaNWumhx9+2Hyu1Jdffqn+/ftr8ODB2r9/v1atWqXt27crJSXlFr+b63DDxPoaGz9+vBESEmJs2bLFyM/PN5fz588bhmEYx44dM2bMmGHs3r3byM3NNf7xj38YrVu3Nu6//35zjMuXLxsdOnQw+vXrZ+Tk5BgZGRlGs2bNjKlTp5o1X331lREUFGRMnjzZOHz4sLFo0SLD19fXyMjIMGveeustw9/f30hPTzc+++wzY9y4cUZoaKjT3YZPPfWU0bJlS2PTpk3G7t27jfj4eCM+Pr7G79eVuwqA2mDYsGHGgw8+aDz44IPGsGHDPN0OgFvsZu4KvNZd/Z9++qkhyTh79qxhGIaxefNmQ5Lx1ltvmTUnT540AgMDjVWrVhmGYRhjxowxxo0b5zTORx99ZPj4+Jj9xcTEGPPmzXO5V8Ow7q5Aj86xWrx4saTv7hy40tKlSzVy5Ej5+fnpn//8p+bPn6/S0lJFR0dr8ODBeuGFF8xaX19frVu3TuPHj1d8fLwaNGigESNGaMaMGWZNbGys3n//fU2aNEkLFixQixYt9MYbb8hut5s1Q4YMUVFRkdLS0uRwONSlSxdlZGQ4TWifN2+efHx8NHjwYJWVlclut+u1115z06cDAEDdkJ2drenTp2vfvn06ffq0KisrJX33gO+qq0uSnK4ANWnSRG3bttXhw4clSfv27dP+/fu1YsUKs8YwDFVWVio3N1d33nnnLXo31+bRYGUYxjX3R0dHa+vWrdcdJyYmRh988ME1a3r37q29e/desyYlJeWapxQDAgK0aNEiLVq06Lo9AQAAqbS0VHa7XXa7XStWrFCzZs2Ul5cnu92u8vLyGo9z7tw5Pfnkk3rmmWeq7WvZsqWVLd+UWnNXIAAAqHuOHDmikydPavbs2eYjh3bv3n3V2p07d5oh6fTp0/r888/NM1HdunXTZ599pjZt2tyaxm9QrbgrEAAA1E0tW7aUn5+f/vznP+urr77Su+++q5kzZ161dsaMGcrMzNTBgwc1cuRIhYWFKSkpSZL03HPPaceOHUpJSVFOTo6++OIL/eMf/6h1k9cJVgAAwG2aNWum9PR0rV69Wu3bt9fs2bP18ssvX7V29uzZmjBhguLi4uRwOPTee+/Jz89PktSpUydt3bpVn3/+uXr16qWuXbsqLS1NUVFRt/LtXJfNuN5EJ1impKREISEhKi4uVnBwsKfbAa7rscceM5/vFhERoZUrV3q4IwC30sWLF5Wbm6vY2FgFBAR4uh23utZ7deXfb85YAQAAWIRgBQAAYBGCFQAAgEUIVgAAABYhWAEAAFiEYAUAAGARghUAAIBFCFYAAAAWIVgBAABYhB9hBgAALoubvPyWHi/7pcdv6HWLFi3SSy+9JIfDoc6dO+vPf/6zevToYXF3/8EZKwAAUCetWrVKqampmjZtmvbs2aPOnTvLbrersLDQbcckWAEAgDpp7ty5Gjt2rEaNGqX27dtryZIlCgoK0ptvvum2Y3IpELXKhAkTVFRUJOm7X0RfsGCBhzsCAHij8vJyZWdna+rUqeY2Hx8fJSQkKCsry23HJVihVikqKlJBQYGn2wAAeLlvv/1WFRUVioiIcNoeERGhI0eOuO24XAoEAACwCMEKAADUOWFhYfL19a12FaSgoECRkZFuOy7BCgAA1Dl+fn6Ki4tTZmamua2yslKZmZmKj49323GZYwUAAOqk1NRUjRgxQt27d1ePHj00f/58lZaWatSoUW47JsEKAADUSUOGDFFRUZHS0tLkcDjUpUsXZWRkVJvQbiWCFQAAcNmNPgn9VktJSVFKSsotOx5zrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIjzHqg6Km7zc0y3csODT58y0n3/6nFe/F8l7nvMCALAGZ6wAAAAsQrACAACwCJcCAQCAy/JmdLylx2uZdsCl+m3btumll15Sdna28vPztWbNGiUlJbmnuStwxgoAANQ5paWl6ty5sxYtWnRLj8sZK8CNbvV/0Vnt8pmmknz//fcJr34/rv7XLgDvNmDAAA0YMOCWH5czVgAAABYhWAEAAFiEYAUAAGARghUAAIBFCFYAAAAW4a5AAABQ55w7d07Hjh0z13Nzc5WTk6MmTZqoZcuWbjsuwQoAANQ5u3fvVp8+fcz11NRUSdKIESOUnp7utuMSrAAAgMtq+7PhevfuLcMwbvlxCVaoVSrrN7jq3wAAeAOCFWqVc21v/VNyAQCwCncFAgAAWIRgBQAAYBGCFQAAuCZPTAK/1ax6j8yxAgDgNjRhwgQVFRVJkpo1a6YFCxZUq6lfv74k6fz58woMDLyl/d1q58+fl/Sf93yjCFYAANyGioqKVFBQcM0aX19fhYaGqrCwUJIUFBQkm812K9q7ZQzD0Pnz51VYWKjQ0FD5+vre1HgEKwAA8L0iIyMlyQxXdVVoaKj5Xm8GwQoAAHwvm82m5s2bKzw8XJcuXfJ0O25Rv379mz5TVYVgBQAArsvX19ey8FGXcVcgAACARQhWAAAAFiFYAQAAWIRgBQAAYBGCFQAAgEU8GqxmzZqlu+++W40aNVJ4eLiSkpJ09OhRp5qLFy8qOTlZTZs2VcOGDTV48OBqDzTLy8tTYmKigoKCFB4ersmTJ+vy5ctONVu2bFG3bt3k7++vNm3aKD09vVo/ixYtUqtWrRQQEKCePXtq165dLvcCAABuXx4NVlu3blVycrJ27typjRs36tKlS+rXr59KS0vNmkmTJum9997T6tWrtXXrVp04cUKDBg0y91dUVCgxMVHl5eXasWOHli1bpvT0dKWlpZk1ubm5SkxMVJ8+fZSTk6OJEyfqiSee0Pr1682aVatWKTU1VdOmTdOePXvUuXNn2e12pweiXa8XAABwe7MZteiXFYuKihQeHq6tW7fq/vvvV3FxsZo1a6aVK1fqpz/9qSTpyJEjuvPOO5WVlaV77rlHH374oX7yk5/oxIkTioiIkCQtWbJEzz33nIqKiuTn56fnnntO77//vg4ePGgea+jQoTpz5owyMjIkST179tTdd9+thQsXSpIqKysVHR2tp59+Ws8//3yNevlvZWVlKisrM9dLSkoUHR2t4uJiBQcHu+dDlBQ3ebnbxoZr1jR6ydMt3JRf72yqk2XfPbemqX+FXr7npIc7unEt0w54ugWgVnnsscfMqy4RERFauXKlhzuqvUpKShQSElKjf79r1Ryr4uJiSVKTJk0kSdnZ2bp06ZISEhLMmnbt2qlly5bKysqSJGVlZaljx45mqJIku92ukpISHTp0yKy5coyqmqoxysvLlZ2d7VTj4+OjhIQEs6Ymvfy3WbNmKSQkxFyio6Nv7IMBAABeodYEq8rKSk2cOFH33nuvOnToIElyOBzy8/NTaGioU21ERIQcDodZc2Woqtpfte9aNSUlJbpw4YK+/fZbVVRUXLXmyjGu18t/mzp1qoqLi83lm2++qeGnAQAAvFGt+Umb5ORkHTx4UNu3b/d0K5bx9/eXv7+/p9sAAAC3SK04Y5WSkqJ169Zp8+bNatGihbk9MjJS5eXlOnPmjFN9QUGB+QvUkZGR1e7Mq1q/Xk1wcLACAwMVFhYmX1/fq9ZcOcb1egEAALc3jwYrwzCUkpKiNWvWaNOmTYqNjXXaHxcXp/r16yszM9PcdvToUeXl5Sk+Pl6SFB8frwMHDjjdvbdx40YFBwerffv2Zs2VY1TVVI3h5+enuLg4p5rKykplZmaaNTXpBQAA3N48eikwOTlZK1eu1D/+8Q81atTInKsUEhKiwMBAhYSEaMyYMUpNTVWTJk0UHBysp59+WvHx8eZdeP369VP79u31y1/+UnPmzJHD4dALL7yg5ORk8zLcU089pYULF2rKlCkaPXq0Nm3apLffflvvv/++2UtqaqpGjBih7t27q0ePHpo/f75KS0s1atQos6fr9QIAAG5vHg1WixcvliT17t3bafvSpUs1cuRISdK8efPk4+OjwYMHq6ysTHa7Xa+99ppZ6+vrq3Xr1mn8+PGKj49XgwYNNGLECM2YMcOsiY2N1fvvv69JkyZpwYIFatGihd544w3Z7XazZsiQISoqKlJaWpocDoe6dOmijIwMpwnt1+sFAADc3mrVc6zqOleeg3EzeI5V7cFzrGoPnmMFOOM5VjXntc+xAgAA8GYEKwAAAIsQrAAAACxSax4QCqD2aeJfcdW/AQBXR7AC8L1+0/WMp1sAAK/CpUAAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIvw5HUAAG5A3oyOnm7hplw+01SS77//PuHV76dl2gFPt2DijBUAAIBFCFYAAAAWIVgBAABYhGAFAABgEYIVAACARbgrEABwS0yYMEFFRUWSpGbNmmnBggUe7giwHsEKAHBLFBUVqaCgwNNtAG7FpUAAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACzCT9oAgBeJm7zc0y3csODT58z/ms8/fc6r34skrWnk6Q5QG3HGCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIjccrI4dO6b169frwoULkiTDMCxrCgAAwBu5HKxOnjyphIQE3XHHHXrooYeUn58vSRozZoyeffZZyxsEANQNlfUbqNLv30v9Bp5uB3ALl4PVpEmTVK9ePeXl5SkoKMjcPmTIEGVkZFjaHACg7jjXdoBKOvxUJR1+qnNtB3i6ndteE/8KNf330sS/wtPt1Bku/wjzhg0btH79erVo0cJp+49+9CP961//sqwxAADgPr/pesbTLdRJLp+xKi0tdTpTVeXUqVPy9/e3pCkAAABv5HKw6tWrl5YvX26u22w2VVZWas6cOerTp4+lzQEAAHgTly8FzpkzR3379tXu3btVXl6uKVOm6NChQzp16pQ+/vhjd/QIAADgFVw+Y9WhQwd9/vnnuu+++/TII4+otLRUgwYN0t69e/XDH/7QHT0CAAB4BZfPWElSSEiIfvvb31rdCwAAgFe7oWB18eJF7d+/X4WFhaqsrHTaN3DgQEsaAwAA8DYuB6uMjAw9/vjj+vbbb6vts9lsqqjgWRgAAOD25PIcq6efflo/+9nPlJ+fr8rKSqeFUAUAAG5nLgergoICpaamKiIiwh39AAAAeC2Xg9VPf/pTbdmyxQ2tAAAAeDeX51gtXLhQP/vZz/TRRx+pY8eOql+/vtP+Z555xrLmAAAAvInLwervf/+7NmzYoICAAG3ZskU2m83cZ7PZCFYAAOC25XKw+u1vf6sXX3xRzz//vHx8XL6SCAAAUGe5nIzKy8s1ZMgQQhUAAMB/cTkdjRgxQqtWrbLk4Nu2bdPDDz+sqKgo2Ww2rV271mn/yJEjZbPZnJb+/fs71Zw6dUrDhw9XcHCwQkNDNWbMGJ07d86pZv/+/erVq5cCAgIUHR2tOXPmVOtl9erVateunQICAtSxY0d98MEHTvsNw1BaWpqaN2+uwMBAJSQk6IsvvrDkcwAAAHWDy5cCKyoqNGfOHK1fv16dOnWqNnl97ty5NR6rtLRUnTt31ujRozVo0KCr1vTv319Lly411/39/Z32Dx8+XPn5+dq4caMuXbqkUaNGady4cVq5cqUkqaSkRP369VNCQoKWLFmiAwcOaPTo0QoNDdW4ceMkSTt27NCwYcM0a9Ys/eQnP9HKlSuVlJSkPXv2qEOHDpK++/HpV199VcuWLVNsbKx+97vfyW6367PPPlNAQECN3zMAAKi7XA5WBw4cUNeuXSVJBw8edNp35UT2mhgwYIAGDBhwzRp/f39FRkZedd/hw4eVkZGhTz/9VN27d5ck/fnPf9ZDDz2kl19+WVFRUVqxYoXKy8v15ptvys/PT3fddZdycnI0d+5cM1gtWLBA/fv31+TJkyVJM2fO1MaNG7Vw4UItWbJEhmFo/vz5euGFF/TII49IkpYvX66IiAitXbtWQ4cOdel9AwCAusnlYLV582Z39PG9tmzZovDwcDVu3FgPPvigfv/736tp06aSpKysLIWGhpqhSpISEhLk4+OjTz75RI8++qiysrJ0//33y8/Pz6yx2+3605/+pNOnT6tx48bKyspSamqq03Htdrt5aTI3N1cOh0MJCQnm/pCQEPXs2VNZWVnfG6zKyspUVlZmrpeUlNz05wEAAGqvm5qBfvz4cR0/ftyqXqrp37+/li9frszMTP3pT3/S1q1bNWDAAPOncxwOh8LDw51eU69ePTVp0kQOh8Os+e+nxFetX6/myv1Xvu5qNVcza9YshYSEmEt0dLRL7x8AAHgXl4NVZWWlZsyYoZCQEMXExCgmJkahoaGaOXOmKisrLW1u6NChGjhwoDp27KikpCStW7dOn376qdc8+X3q1KkqLi42l2+++cbTLQEAADe6oedY/fWvf9Xs2bN17733SpK2b9+u6dOn6+LFi/rDH/5geZNVWrdurbCwMB07dkx9+/ZVZGSkCgsLnWouX76sU6dOmfOyIiMjVVBQ4FRTtX69miv3V21r3ry5U02XLl2+t19/f/9qk+0BAEDd5fIZq2XLlumNN97Q+PHj1alTJ3Xq1Em/+tWv9Je//EXp6eluaPE/jh8/rpMnT5rhJj4+XmfOnFF2drZZs2nTJlVWVqpnz55mzbZt23Tp0iWzZuPGjWrbtq0aN25s1mRmZjoda+PGjYqPj5ckxcbGKjIy0qmmpKREn3zyiVkDAADgcrA6deqU2rVrV217u3btdOrUKZfGOnfunHJycpSTkyPpu0niOTk5ysvL07lz5zR58mTt3LlTX3/9tTIzM/XII4+oTZs2stvtkqQ777xT/fv319ixY7Vr1y59/PHHSklJ0dChQxUVFSVJeuyxx+Tn56cxY8bo0KFDWrVqlRYsWOA0WX3ChAnKyMjQK6+8oiNHjmj69OnavXu3UlJSJH13t+PEiRP1+9//Xu+++64OHDigxx9/XFFRUUpKSnL1IwQAAHWUy8Gqc+fOWrhwYbXtCxcuVOfOnV0aa/fu3eratav5+IbU1FR17dpVaWlp8vX11f79+zVw4EDdcccdGjNmjOLi4vTRRx85XV5bsWKF2rVrp759++qhhx7Sfffdp9dff93cHxISog0bNig3N1dxcXF69tlnlZaWZj5qQZJ+/OMfa+XKlXr99dfVuXNn/d///Z/Wrl1rPsNKkqZMmaKnn35a48aN0913361z584pIyODZ1gBAACTzTAMw5UXbN26VYmJiWrZsqV5GSwrK0vffPONPvjgA/Xq1cstjdYFJSUlCgkJUXFxsYKDg912nLjJy902NlyzptFLnm4B/9Yy7YCnW7AE3+/ag+937eHu77cr/367fMbqgQce0NGjR/Xoo4/qzJkzOnPmjAYNGqSjR48SqgAAwG3N5bsCJekHP/iBW+/+AwAA8EYun7FaunSpVq9eXW376tWrtWzZMkuaAgAA8EYuB6tZs2YpLCys2vbw8HD98Y9/tKQpAAAAb+RysMrLy1NsbGy17TExMcrLy7OkKQAAAG/kcrAKDw/X/v37q23ft2+f+ePIAAAAtyOXg9WwYcP0zDPPaPPmzaqoqFBFRYU2bdqkCRMmaOjQoe7oEQAAwCu4fFfgzJkz9fXXX6tv376qV++7l1dWVurxxx9njhUAALituRys/Pz8tGrVKs2cOVP79u1TYGCgOnbsqJiYGHf0BwAA4DVu6DlWknTHHXfojjvusLIXAAAAr+ZysKqoqFB6eroyMzNVWFioyspKp/2bNm2yrDkAAABv4nKwmjBhgtLT05WYmKgOHTrIZrO5oy8AAACv43Kweuutt/T222/roYceckc/AAAAXsvlxy34+fmpTZs27ugFAADAq7kcrJ599lktWLBAhmG4ox8AAACv5fKlwO3bt2vz5s368MMPddddd6l+/fpO+9955x3LmgMAAPAmLger0NBQPfroo+7oBQAAwKu5HKyWLl3qjj4AAAC8nstzrAAAAHB1NTpj1a1bN2VmZqpx48bq2rXrNZ9dtWfPHsuaAwAA8CY1ClaPPPKI/P39JUlJSUnu7AcAAMBr1ShYTZs27ap/X8vf//53DRw4UA0aNLixzgAAALyM2+ZYPfnkkyooKHDX8AAAALWO24IVDxAFAAC3G+4KBAAAsAjBCgAAwCIEKwAAAIsQrAAAACzitmAVExNT7QeaAQAA6jKXfyuwpg4ePOiuoQEAAGqlGgWrxo0bX/NnbK506tSpm2oIAADAW9UoWM2fP9/NbQAAAHi/GgWrESNGuLsPAAAAr3dTc6wuXryo8vJyp23BwcE31RAAAIC3cvmuwNLSUqWkpCg8PFwNGjRQ48aNnRYAAIDblcvBasqUKdq0aZMWL14sf39/vfHGG3rxxRcVFRWl5cuXu6NHAAAAr+DypcD33ntPy5cvV+/evTVq1Cj16tVLbdq0UUxMjFasWKHhw4e7o08AAIBaz+UzVqdOnVLr1q0lfTefqurxCvfdd5+2bdtmbXcAAABexOVg1bp1a+Xm5kqS2rVrp7ffflvSd2eyQkNDLW0OAADAm7gcrEaNGqV9+/ZJkp5//nktWrRIAQEBmjRpkiZPnmx5gwAAAN7C5TlWkyZNMv9OSEjQkSNHlJ2drTZt2qhTp06WNgcAAOBNbvq3AmNiYhQTE2NFLwAAAF7N5WA1Y8aMa+5PS0u74WYAAAC8mcvBas2aNU7rly5dUm5ururVq6cf/vCHBCsAAHDbcjlY7d27t9q2kpISjRw5Uo8++qglTQEAAHgjl+8KvJrg4GC9+OKL+t3vfmfFcAAAAF7JkmAlScXFxSouLrZqOAAAAK/j8qXAV1991WndMAzl5+frb3/7mwYMGGBZYwAAAN7G5WA1b948p3UfHx81a9ZMI0aM0NSpUy1rDAAAwNu4HKyqfs4GAAAAziybYwUAAHC7c/mMVWlpqWbPnq3MzEwVFhaqsrLSaf9XX31lWXMAAADexOVg9cQTT2jr1q365S9/qebNm8tms7mjLwAAAK/jcrD68MMP9f777+vee+91Rz8AAABey+U5Vo0bN1aTJk3c0QsAAIBXczlYzZw5U2lpaTp//rw7+gEAAPBaLl8KfOWVV/Tll18qIiJCrVq1Uv369Z3279mzx7LmAAAAvInLwSopKckNbQAAAHg/l4PVtGnT3NEHAACA13M5WFUpLy+/6nOsWrZsedNNAQAAeCOXJ69//vnn6tWrlwIDAxUTE6PY2FjFxsaqVatWio2NdWmsbdu26eGHH1ZUVJRsNpvWrl3rtN8wDKWlpal58+YKDAxUQkKCvvjiC6eaU6dOafjw4QoODlZoaKjGjBmjc+fOOdXs379fvXr1UkBAgKKjozVnzpxqvaxevVrt2rVTQECAOnbsqA8++MDlXgAAwO3N5WA1atQo+fj4aN26dcrOztaePXu0Z88e7d271+WJ66WlpercubMWLVp01f1z5szRq6++qiVLluiTTz5RgwYNZLfbdfHiRbNm+PDhOnTokDZu3Kh169Zp27ZtGjdunLm/pKRE/fr1U0xMjLKzs/XSSy9p+vTpev31182aHTt2aNiwYRozZoz27t2rpKQkJSUl6eDBgy71AgAAbm82wzAMV17QoEEDZWdnq127dtY2YrNpzZo15uR4wzAUFRWlZ599Vr/+9a8lScXFxYqIiFB6erqGDh2qw4cPq3379vr000/VvXt3SVJGRoYeeughHT9+XFFRUVq8eLF++9vfyuFwyM/PT5L0/PPPa+3atTpy5IgkaciQISotLdW6devMfu655x516dJFS5YsqVEvNVFSUqKQkBAVFxcrODjYks/tauImL3fb2HDNmkYveboF/FvLtAOebsESfL9rD77ftYe7v9+u/Pvt8hmr9u3b69tvv73h5moqNzdXDodDCQkJ5raQkBD17NlTWVlZkqSsrCyFhoaaoUqSEhIS5OPjo08++cSsuf/++81QJUl2u11Hjx7V6dOnzZorj1NVU3WcmvRyNWVlZSopKXFaAABA3eVysPrTn/6kKVOmaMuWLTp58qTbgoPD4ZAkRUREOG2PiIgw9zkcDoWHhzvtr1evnpo0aeJUc7UxrjzG99Vcuf96vVzNrFmzFBISYi7R0dHXedcAAMCbuXxXYNVZm759+zptNwxDNptNFRUV1nRWB0ydOlWpqanmeklJCeEKAIA6zOVgtXnz5u/dd+CAddc4IyMjJUkFBQVq3ry5ub2goEBdunQxawoLC51ed/nyZZ06dcp8fWRkpAoKCpxqqtavV3Pl/uv1cjX+/v7y9/ev0fsFAADez+VLgQ888IDT0q1bNx09elSTJ0/WhAkTLGssNjZWkZGRyszMNLeVlJTok08+UXx8vCQpPj5eZ86cUXZ2tlmzadMmVVZWqmfPnmbNtm3bdOnSJbNm48aNatu2rRo3bmzWXHmcqpqq49SkFwAAAJeDVZVt27ZpxIgRat68uV5++WU9+OCD2rlzp0tjnDt3Tjk5OcrJyZH03STxnJwc5eXlyWazaeLEifr973+vd999VwcOHNDjjz+uqKgo887BO++8U/3799fYsWO1a9cuffzxx0pJSdHQoUMVFRUlSXrsscfk5+enMWPG6NChQ1q1apUWLFjgdIluwoQJysjI0CuvvKIjR45o+vTp2r17t1JSUiSpRr0AAAC4dCnQ4XAoPT1df/3rX1VSUqKf//znKisr09q1a9W+fXuXD75792716dPHXK8KOyNGjFB6erqmTJmi0tJSjRs3TmfOnNF9992njIwMBQQEmK9ZsWKFUlJS1LdvX/n4+Gjw4MF69dVXzf0hISHasGGDkpOTFRcXp7CwMKWlpTk96+rHP/6xVq5cqRdeeEG/+c1v9KMf/Uhr165Vhw4dzJqa9AIAAG5vNX6O1cMPP6xt27YpMTFRw4cPV//+/eXr66v69etr3759NxSsbjc8x+r2w3Nuag+eYwWr8f2uPWrTc6xqfMbqww8/1DPPPKPx48frRz/60U03CQAAUNfUeI7V9u3bdfbsWcXFxalnz55auHDhLXlQKAAAgLeocbC655579Je//EX5+fl68skn9dZbbykqKkqVlZXauHGjzp49684+AQAAaj2X7wps0KCBRo8ere3bt+vAgQN69tlnNXv2bIWHh2vgwIHu6BEAAMAr3PDjFiSpbdu2mjNnjo4fP66///3vVvUEAADglW4qWFXx9fVVUlKS3n33XSuGAwAA8EqWBCsAAAAQrAAAACxDsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsUuuD1fTp02Wz2ZyWdu3amfsvXryo5ORkNW3aVA0bNtTgwYNVUFDgNEZeXp4SExMVFBSk8PBwTZ48WZcvX3aq2bJli7p16yZ/f3+1adNG6enp1XpZtGiRWrVqpYCAAPXs2VO7du1yy3sGAADeqdYHK0m66667lJ+fby7bt283902aNEnvvfeeVq9era1bt+rEiRMaNGiQub+iokKJiYkqLy/Xjh07tGzZMqWnpystLc2syc3NVWJiovr06aOcnBxNnDhRTzzxhNavX2/WrFq1SqmpqZo2bZr27Nmjzp07y263q7Cw8NZ8CAAAoNbzimBVr149RUZGmktYWJgkqbi4WH/96181d+5cPfjgg4qLi9PSpUu1Y8cO7dy5U5K0YcMGffbZZ/rf//1fdenSRQMGDNDMmTO1aNEilZeXS5KWLFmi2NhYvfLKK7rzzjuVkpKin/70p5o3b57Zw9y5czV27FiNGjVK7du315IlSxQUFKQ333zz1n8gAACgVvKKYPXFF18oKipKrVu31vDhw5WXlydJys7O1qVLl5SQkGDWtmvXTi1btlRWVpYkKSsrSx07dlRERIRZY7fbVVJSokOHDpk1V45RVVM1Rnl5ubKzs51qfHx8lJCQYNZcTVlZmUpKSpwWAABQd9X6YNWzZ0+lp6crIyNDixcvVm5urnr16qWzZ8/K4XDIz89PoaGhTq+JiIiQw+GQJDkcDqdQVbW/at+1akpKSnThwgV9++23qqiouGpN1RhXM2vWLIWEhJhLdHT0DX0GAADAO9TzdAPXM2DAAPPvTp06qWfPnoqJidHbb7+twMBAD3Z2fVOnTlVqaqq5XlJSQrgCAKAOq/VnrP5baGio7rjjDh07dkyRkZEqLy/XmTNnnGoKCgoUGRkpSYqMjKx2l2DV+vVqgoODFRgYqLCwMPn6+l61pmqMq/H391dwcLDTAgAA6i6vC1bnzp3Tl19+qebNmysuLk7169dXZmamuf/o0aPKy8tTfHy8JCk+Pl4HDhxwuntv48aNCg4OVvv27c2aK8eoqqkaw8/PT3FxcU41lZWVyszMNGsAAABqfbD69a9/ra1bt+rrr7/Wjh079Oijj8rX11fDhg1TSEiIxowZo9TUVG3evFnZ2dkaNWqU4uPjdc8990iS+vXrp/bt2+uXv/yl9u3bp/Xr1+uFF15QcnKy/P39JUlPPfWUvvrqK02ZMkVHjhzRa6+9prfffluTJk0y+0hNTdVf/vIXLVu2TIcPH9b48eNVWlqqUaNGeeRzAQAAtU+tn2N1/PhxDRs2TCdPnlSzZs103333aefOnWrWrJkkad68efLx8dHgwYNVVlYmu92u1157zXy9r6+v1q1bp/Hjxys+Pl4NGjTQiBEjNGPGDLMmNjZW77//viZNmqQFCxaoRYsWeuONN2S3282aIUOGqKioSGlpaXI4HOrSpYsyMjKqTWgHAAC3L5thGIanm7hdlJSUKCQkRMXFxW6dbxU3ebnbxoZr1jR6ydMt4N9aph3wdAuW4Ptde/D9rj3c/f125d/vWn8pEAAAwFsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACxCsAIAALAIwQoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKxctWrRIrVq1UkBAgHr27Kldu3Z5uiUAAFBLEKxcsGrVKqWmpmratGnas2ePOnfuLLvdrsLCQk+3BgAAagGClQvmzp2rsWPHatSoUWrfvr2WLFmioKAgvfnmm55uDQAA1AL1PN2AtygvL1d2dramTp1qbvPx8VFCQoKysrKu+pqysjKVlZWZ68XFxZKkkpISt/ZaUXbBreOj5s7Wr/B0C/g3d3/vbhW+37UH3+/aw93f76rxDcO4bi3Bqoa+/fZbVVRUKCIiwml7RESEjhw5ctXXzJo1Sy+++GK17dHR0W7pEbVPB083gP+YFeLpDlDH8P2uRW7R9/vs2bMKCbn2sQhWbjR16lSlpqaa65WVlTp16pSaNm0qm83mwc5wK5SUlCg6OlrffPONgoODPd0OAAvx/b69GIahs2fPKioq6rq1BKsaCgsLk6+vrwoKCpy2FxQUKDIy8qqv8ff3l7+/v9O20NBQd7WIWio4OJj/4wXqKL7ft4/rnamqwuT1GvLz81NcXJwyMzPNbZWVlcrMzFR8fLwHOwMAALUFZ6xckJqaqhEjRqh79+7q0aOH5s+fr9LSUo0aNcrTrQEAgFqAYOWCIUOGqKioSGlpaXI4HOrSpYsyMjKqTWgHpO8uBU+bNq3a5WAA3o/vN76PzajJvYMAAAC4LuZYAQAAWIRgBQAAYBGCFQAAgEUIVgAAABYhWAEWGTlypGw2m5566qlq+5KTk2Wz2TRy5Mhb3xgAy1R9z/97OXbsmKdbQy1BsAIsFB0drbfeeksXLvznh3IvXryolStXqmXLlh7sDIBV+vfvr/z8fKclNjbW022hliBYARbq1q2boqOj9c4775jb3nnnHbVs2VJdu3b1YGcArOLv76/IyEinxdfX19NtoZYgWAEWGz16tJYuXWquv/nmmzydHwBuEwQrwGK/+MUvtH37dv3rX//Sv/71L3388cf6xS9+4em2AFhk3bp1atiwobn87Gc/83RLqEX4SRvAYs2aNVNiYqLS09NlGIYSExMVFhbm6bYAWKRPnz5avHixud6gQQMPdoPahmAFuMHo0aOVkpIiSVq0aJGHuwFgpQYNGqhNmzaebgO1FMEKcIP+/furvLxcNptNdrvd0+0AAG4RghXgBr6+vjp8+LD5NwDg9kCwAtwkODjY0y0AAG4xm2EYhqebAAAAqAt43AIAAIBFCFYAAAAWIVgBAABYhGAFAABgEYIVAACARQhWAAAAFiFYAQAAWIRgBQAAYBGCFQAAgEUIVgDqNIfDoQkTJqhNmzYKCAhQRESE7r33Xi1evFjnz5+XJLVq1Uo2m63aMnv2bEnS119/LZvNpvDwcJ09e9Zp/C5dumj69Onmeu/evc3X+/v76wc/+IEefvhhvfPOO9V6u9oxbTab3nrrLUnSli1bnLY3a9ZMDz30kA4cOOCmTwvAzSJYAaizvvrqK3Xt2lUbNmzQH//4R+3du1dZWVmaMmWK1q1bp3/+859m7YwZM5Sfn++0PP30007jnT17Vi+//PJ1jzt27Fjl5+fryy+/1P/7f/9P7du319ChQzVu3LhqtUuXLq123KSkJKeao0ePKj8/X+vXr1dZWZkSExNVXl5+Yx8KALfiR5gB1Fm/+tWvVK9ePe3evVsNGjQwt7du3VqPPPKIrvyp1EaNGikyMvKa4z399NOaO3eukpOTFR4e/r11QUFB5lgtWrTQPffco3bt2mn06NH6+c9/roSEBLM2NDT0uscNDw836yZOnKiBAwfqyJEj6tSp0zVfB+DW44wVgDrp5MmT2rBhg5KTk51C1ZVsNptLYw4bNkxt2rTRjBkzXO5nxIgRaty48VUvCdZUcXGxeZnQz8/vhscB4D4EKwB10rFjx2QYhtq2beu0PSwsTA0bNlTDhg313HPPmdufe+45c3vV8tFHHzm9tmre1euvv64vv/zSpX58fHx0xx136Ouvv3baPmzYsGrHzcvLc6pp0aKFGjZsqNDQUK1cuVIDBw5Uu3btXDo+gFuDS4EAbiu7du1SZWWlhg8frrKyMnP75MmTNXLkSKfaH/zgB9Veb7fbdd999+l3v/udVq5c6dKxDcOodpZs3rx5TpcGJSkqKspp/aOPPlJQUJB27typP/7xj1qyZIlLxwVw6xCsANRJbdq0kc1m09GjR522t27dWpIUGBjotD0sLExt2rSp0dizZ89WfHy8Jk+eXON+Kioq9MUXX+juu+922h4ZGXnd48bGxio0NFRt27ZVYWGhhgwZom3bttX42ABuHS4FAqiTmjZtqv/5n//RwoULVVpaaunYPXr00KBBg/T888/X+DXLli3T6dOnNXjw4Js6dnJysg4ePKg1a9bc1DgA3IMzVgDqrNdee0333nuvunfvrunTp6tTp07y8fHRp59+qiNHjiguLs6sPXv2rBwOh9Prg4KCFBwcfNWx//CHP+iuu+5SvXrV/2/0/Pnzcjgcunz5so4fP641a9Zo3rx5Gj9+vPr06eNUe+bMmWrHbdSo0fdOuA8KCtLYsWM1bdo0JSUluTwBH4CbGQBQh504ccJISUkxYmNjjfr16xsNGzY0evToYbz00ktGaWmpYRiGERMTY0iqtjz55JOGYRhGbm6uIcnYu3ev09jjxo0zJBnTpk0ztz3wwAPm6/38/IzmzZsbP/nJT4x33nmnWm9XO6YkY9asWYZhGMbmzZsNScbp06edXpeXl2fUq1fPWLVqlXUfFABL2Azjige5AAAA4IYxxwoAAMAiBCsAAACLEKwAAAAsQrACAACwCMEKAADAIgQrAAAAixCsAAAALEKwAgAAsAjBCgAAwCIEKwAAAIsQrAAAACzy/wHZFCpV7Oij2gAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X.columns"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X7R7kFcuPQk-",
        "outputId": "d9052c33-69ac-46cd-921e-cb0aa01ac2ab"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['Ind_ID', 'CHILDREN', 'Annual_income', 'Type_Income', 'EDUCATION',\n",
              "       'Housing_type', 'Work_Phone', 'Phone', 'EMAIL_ID', 'Type_Occupation',\n",
              "       'Family_Members', 'age', 'experience', 'Car_Owner_Y', 'Propert_Owner_Y',\n",
              "       'Marital_status_Married', 'Marital_status_Separated',\n",
              "       'Marital_status_Single / not married', 'Marital_status_Widow',\n",
              "       'GENDER_M'],\n",
              "      dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 187
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.barplot(data=credit_card_raw,x='GENDER',y='CHILDREN',hue='label')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 469
        },
        "id": "ubCQeuoTPLPq",
        "outputId": "607a8281-9d54-4c73-feaf-ade38bc73f0e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='GENDER', ylabel='CHILDREN'>"
            ]
          },
          "metadata": {},
          "execution_count": 188
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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QJK0cuVKvf766yooKNDu3bv14Ycf6vnnnx92E4oJNwAAGC4qKkobN27U7373OyUlJWn16tVau3Ztr31Xr16twsJCpaWlyeFw6MUXX/S80HLmzJn63//9X33wwQe6+uqrlZqaqpKSEk2YMGEoD+esbG632211EUPJl1emn88WLlwop9MpSYqJidHmzZstrggAzk+ff/65Dh48qClTpgybu5H85UzH6su/35y5AQAARiHcAAAAoxBuAACAUQg3AADAKIQbAABgFMINAAAwCuEGAAAYhXADAACMQrgBAABGIdwAAACj8FZwAABGqLS7/3NIx2tck3tO21VWVmrNmjVyOBxKTk7WE088oYyMjEGu7mucuQEAAH5TU1OjoqIilZaWateuXUpOTlZ2draam5v9NibhBgAA+E15ebny8/OVl5enpKQkVVVVKSQkRNXV1X4bk3ADAAD8orOzU42NjcrKyvK02e12ZWVlqaGhwW/jEm4AAIBfHD9+XF1dXYqJifFqj4mJkcPh8Nu4hBsAAGAUwg0AAPCLyMhIBQQEyOl0erU7nU7Fxsb6bVxuBQeGucLCQrlcLklSVFSUHnvsMYsrAoD+CQwMVFpamurr67VgwQJJUnd3t+rr61VQUOC3cQk3wDDncrl6/K8HAEaKoqIiLVq0SOnp6crIyFBFRYXa29uVl5fntzEJNwAAwG9ycnLkcrlUUlIih8OhlJQU1dXV9ZhkPJgINwAAjFDn+sTgoVZQUODXy1DfxIRiAABgFMINAAAwCuEGAAAYhXADAACMQrgBAABGIdwAAACjEG4AAIBRCDcAAMAohBsAAGAUwg0AADAKr18AAGCEalp1+ZCON6nkPZ/6v/rqq1qzZo0aGxt17Ngxbd261fN2cH/izA0AAPCL9vZ2JScnq7KyckjH5cwNAADwi+uuu07XXXfdkI/LmRsAAGCUYRFuKisrlZCQoODgYM2ePVs7d+7ss+/GjRtls9m8luDg4CGsFgAADGeWh5uamhoVFRWptLRUu3btUnJysrKzs9Xc3NznNmFhYTp27JhnOXz48BBWDAAAhjPLw015ebny8/OVl5enpKQkVVVVKSQkRNXV1X1uY7PZFBsb61liYmKGsGIAADCcWRpuOjs71djYqKysLE+b3W5XVlaWGhoa+tzu1KlTmjx5suLj43XTTTdp7969ffbt6OhQa2ur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5Zdfavfu3dq0aZMmT548mPUBAAD4xOdw09LSopUrVyoxMVF79+5VfX29XnzxRV122WX+qA8AAMAnPl2W+uUvf6lf/OIXio2N1W9/+9teL1MBAABYyadwc88992jMmDFKTEzUpk2btGnTpl771dbWDkpxAAAAvvIp3OTm5p71VnAAAAAr+RRuNm7c6KcyAAAABodP4aY/mpubFR0dPdi7BQCgh6ZVl1tdwoB8efIiSQH///noiD6eSSXvWV2Ch093S4WEhMjlcnm+z58/X8eOHfN8dzqdGj9+/OBVBwAA4COfws3nn38ut9vt+f7qq6/qs88+8+rz9+sBAACG2qC/OJMJxwAAwEq8FRwAABjFp3Bjs9m8zsx88zsAAIDVfLpbyu1269JLL/UEmlOnTik1NVV2u92zHgAwvHWPvqDXz4ApfAo3GzZs8FcdAIAhcmradVaXAPiVT+Fm0aJF/qoDAABgUAzqhOJjx46poKBgMHcJAADgE5+fULx371698sorCgwM1C233KKIiAgdP35cDz/8sKqqqjR16lR/1AkAANAvPp25eeGFF5SamqoVK1Zo6dKlSk9P1yuvvKIZM2bo/fff19atW7V3715/1QoAAHBWPoWbhx56SMuWLVNra6vKy8t14MABrVixQtu2bVNdXZ3mzZvnrzoBAAD6xadws3//fi1btkyhoaFavny57Ha71q1bp1mzZvmrPgAAAJ/4FG7a2toUFhYmSQoICNCYMWOYYwMAAIYVnycUv/TSSwoPD5ckdXd3q76+Xnv27PHqc+ONNw5OdQAAAD7yOdx881k3t99+u9d3m82mrq6ugVUFAABwjnwKN93d3f6qAwAAYFDwVnAAAGAUny9LSdIf//hH1dbW6tChQ7LZbJoyZYp++MMf6jvf+c5g1wcAAOATn8/cLF26VFlZWfrtb3+rTz/9VC6XS88++6yuueYaLV++/JyKqKysVEJCgoKDgzV79mzt3Lmzz761tbVKT09XRESELrjgAqWkpOiZZ545p3EBAIB5fAo3W7du1YYNG1RdXa3jx4+roaFBb7zxhlwul55++mk99dRTeuGFF3wqoKamRkVFRSotLdWuXbuUnJys7OxsNTc399r/wgsv1L333quGhga9++67ysvLU15enl566SWfxgUAAGbyKdxs2LBBRUVFWrx4sWw229c7sdu1ZMkS3XXX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IiLOOm50dLSn31133aUbb7xR+/bt08yZM8+4HYChxZkbAH716aef6uWXX9ayZcu8gs3fs9lsPu3z1ltvVWJiolatWuVzPYsWLdK4ceN6vTzVXy0tLZ5LVoGBgee8HwD+QbgB4FcfffSR3G63pk2b5tUeGRmp0NBQhYaGauXKlZ72lStXetq/Wl577TWvbb+ah/PUU0/p448/9qkeu92uSy+9VIcOHfJqv/XWW3uM29TU5NVn4sSJCg0NVUREhDZv3qwbb7xR06dP92l8AP7HZSkAlti5c6e6u7t12223qaOjw9N+9913a/HixV594+LiemyfnZ2tf/iHf9B9992nzZs3+zS22+3ucbZo3bp1XpepJGnChAle31977TWFhITojTfe0COPPKKqqiqfxgUwNAg3APwqMTFRNptN+/fv92qfOnWqJGnMmDFe7ZGRkUpMTOzXvlevXq3MzEzdfffd/a6nq6tLH374oWbNmuXVHhsbe9Zxp0yZooiICE2bNk3Nzc3KycnRq6++2u+xAQwNLksB8KuLLrpI1157rZ588km1t7cP6r4zMjL0gx/8QPfcc0+/t9m0aZNOnDihm2++eUBjL1u2THv27NHWrVsHtB8Ag48zNwD87le/+pWuuuoqpaen6/7779fMmTNlt9v15ptvat++fUpLS/P0bWtrk8Ph8No+JCREYWFhve774Ycf1re+9S2NGtXzr7PTp0/L4XDoyy+/1CeffKKtW7dq3bp1uuOOO3TNNdd49T158mSPcceOHdvnJOiQkBDl5+ertLRUCxYs8HlSNAA/cgPAEDh69Ki7oKDAPWXKFPfo0aPdoaGh7oyMDPeaNWvc7e3tbrfb7Z48ebJbUo/l9ttvd7vdbvfBgwfdktxvv/22175/+tOfuiW5S0tLPW1z5szxbB8YGOgeP368+3vf+567tra2R229jSnJXVZW5na73e5XXnnFLcl94sQJr+2amprco0aNctfU1AzeHxSAAbO53X/3gAkAAIARjjk3AADAKIQbAABgFMINAAAwCuEGAAAYhXADAACMQrgBAABGIdwAAACjEG4AAIBRCDcAAMAohBsAAGAUwg0AADAK4QYAABjl/wBMrkXXTNuKfQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.barplot(data=credit_card_raw,x='GENDER',y='age',hue='label')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "REp2P21IPvOA",
        "outputId": "d8cefcf4-553c-4ea6-a609-0ac3a51d9113"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='GENDER', ylabel='age'>"
            ]
          },
          "metadata": {},
          "execution_count": 190
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjIAAAGwCAYAAACzXI8XAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAkBklEQVR4nO3de3BU9d3H8c8m5ELIzQRIwCQSihKQBjCIpIoCxkZEBMkoCrbcxksNCMSqTaugWE0GLRc1gHQQtGMGSgtanBZsowTRRCEKghWUiwabbIJKLgTZpMl5/mjZx20CQtjk7C+8XzM7w/727NnvMhN5e87JrsOyLEsAAAAG8rN7AAAAgNYiZAAAgLEIGQAAYCxCBgAAGIuQAQAAxiJkAACAsQgZAABgrE52D9DWmpqaVFZWprCwMDkcDrvHAQAAZ8GyLNXW1qpnz57y8zv9cZcOHzJlZWWKj4+3ewwAANAKR44cUVxc3Gkf7/AhExYWJuk/fxHh4eE2TwMAAM5GTU2N4uPj3f+On06HD5lTp5PCw8MJGQAADPNDl4VwsS8AADAWIQMAAIxFyAAAAGN1+GtkzlZjY6MaGhrsHqNNBAQEyN/f3+4xAADwugs+ZCzLktPpVFVVld2jtKnIyEjFxsbyWToAgA7lgg+ZUxHTvXt3hYSEdLh/6C3L0okTJ1RZWSlJ6tGjh80TAQDgPbaGzOOPP64nnnjCY61v377at2+fJOnkyZN68MEHtXbtWrlcLqWnp2vZsmWKiYnxyus3Nja6IyY6Otor+/RFnTt3liRVVlaqe/funGYCAHQYtl/se/nll6u8vNx92759u/uxuXPnatOmTVq/fr0KCwtVVlamCRMmeO21T10TExIS4rV9+qpT77GjXgcEALgw2X5qqVOnToqNjW22Xl1drVWrVik/P1+jRo2SJK1evVr9+vVTcXGxhg0b1uL+XC6XXC6X+35NTc0PztDRTie15EJ4jwCAC4/tR2Q+//xz9ezZU71799bkyZNVWloqSSopKVFDQ4PS0tLc2yYlJSkhIUFFRUWn3V9OTo4iIiLcN75nCQCAjsvWkLnqqqu0Zs0abd68WcuXL9fhw4c1fPhw1dbWyul0KjAwUJGRkR7PiYmJkdPpPO0+s7OzVV1d7b4dOXKkjd8FAACwi60hM3r0aN12221KTk5Wenq6/vrXv6qqqkp//OMfW73PoKAg9/cqteX3K40YMUJz5sw5q223bt0qh8Nx3r/i3atXLy1ZsuS89gEAQEdi+6ml74uMjNRll12mAwcOKDY2VvX19c3+8a+oqGjxmhoAAHDh8amQOX78uA4ePKgePXooJSVFAQEBKigocD++f/9+lZaWKjU11cYpAQBondmzZ2vSpEmaNGmSZs+ebfc4HYKtIfPLX/5ShYWF+uKLL/Tee+/p1ltvlb+/v+68805FRERoxowZysrK0ttvv62SkhJNmzZNqampp/2NJbv84Q9/0JAhQxQWFqbY2FhNmjTJ/QF03/fuu+8qOTlZwcHBGjZsmPbu3evx+Pbt2zV8+HB17txZ8fHxeuCBB1RXV9debwMA0MaOHj2qiooKVVRU6OjRo3aP0yHYGjJfffWV7rzzTvXt21e33367oqOjVVxcrG7dukmSFi9erJtvvlkZGRm69tprFRsbqw0bNtg5cosaGhr05JNPavfu3Xrttdf0xRdfaOrUqc22e+ihh/S73/1OO3bsULdu3TR27Fj357ocPHhQN954ozIyMvTxxx9r3bp12r59u2bOnNnO7wYAAHPY+jkya9euPePjwcHBysvLU15eXjtN1DrTp093/7l379567rnndOWVV+r48eMKDQ11PzZ//nzdcMMNkqSXX35ZcXFx2rhxo26//Xbl5ORo8uTJ7guIL730Uj333HO67rrrtHz5cgUHB7frewIAwAQ+dY2MqUpKSjR27FglJCQoLCxM1113nSS5PxPnlO9f2xMVFaW+ffvq008/lSTt3r1ba9asUWhoqPuWnp6upqYmHT58uP3eDAAABrH9k31NV1dXp/T0dKWnp+vVV19Vt27dVFpaqvT0dNXX15/1fo4fP657771XDzzwQLPHEhISvDkyAAAdBiFznvbt26dvvvlGubm57k8R3rlzZ4vbFhcXu6Pk2LFj+uyzz9SvXz9J0hVXXKF//vOf6tOnT/sMDgBAB8CppfOUkJCgwMBAPf/88zp06JD+8pe/6Mknn2xx2wULFqigoEB79+7V1KlT1bVrV40fP16S9Mgjj+i9997TzJkztWvXLn3++ed6/fXXudgXAIAzIGTOU7du3bRmzRqtX79e/fv3V25urp599tkWt83NzdXs2bOVkpIip9OpTZs2KTAwUJKUnJyswsJCffbZZxo+fLgGDx6sefPmqWfPnu35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mDFDDz74oFcHBAAAOJ1WhczcuXPVqVMnlZaWKiQkxL0+ceJEbd682WvDAQAAnEmn1jzpzTff1JYtWxQXF+exfumll+rLL7/0ymAAAAA/pFVHZOrq6jyOxJzy7bffKigo6LyHAgAAOButCpnhw4frlVdecd93OBxqamrSwoULNXLkSK8NBwAAcCatOrW0cOFCXX/99dq5c6fq6+v18MMP65NPPtG3336rd99919szAgAAtKhVR2QGDBigzz77TNdcc43GjRunuro6TZgwQR999JF+9KMfeXtGAACAFrXqiIwkRURE6De/+Y03ZwEAADgnrQqZjz/+uMV1h8Oh4OBgJSQkcNEvAABoc60KmUGDBsnhcEj6/0/zPXVfkgICAjRx4kS9+OKLCg4O9sKYAAAAzbXqGpmNGzfq0ksv1cqVK7V7927t3r1bK1euVN++fZWfn69Vq1bprbfe0qOPPurteQEAANxadUTmqaee0tKlS5Wenu5e+/GPf6y4uDg99thj+uCDD9SlSxc9+OCDevbZZ702LAAAwPe16ojMnj17dMkllzRbv+SSS7Rnzx5J/zn9dOo7mAAAANpCq0ImKSlJubm5qq+vd681NDQoNzdXSUlJkqR//etfiomJ8c6UAAAALWjVqaW8vDzdcsstiouLU3JysqT/HKVpbGzUG2+8IUk6dOiQ7r//fu9NCgAA8D9aFTI/+clPdPjwYb366qv67LPPJEm33XabJk2apLCwMEnSz372M+9NCQAA0IJWfyBeWFiYrr32WvXq1ct9iuntt9+WJN1yyy3emQ4AAOAMWhUyhw4d0q233qo9e/bI4XDIsiyPz5FpbGz02oAAAACn06qLfWfPnq3ExERVVlYqJCREe/fuVWFhoYYMGaKtW7d6eUQAAICWteqITFFRkd566y117dpVfn5+8vf31zXXXKOcnBw98MAD+uijj7w9JwAAQDOtOiLT2Njovqi3a9euKisrk/Sfz5HZv3+/96YDAAA4g1YdkRkwYIB2796txMREXXXVVVq4cKECAwO1cuVK9e7d29szAgAAtKhVIfPoo4+qrq5OkrRgwQLdfPPNGj58uKKjo7Vu3TqvDggAAHA6rQqZ73/HUp8+fbRv3z59++23uuiiizx+ewkAAKAttfpzZP5XVFSUt3YFAABwVlp1sS8AAIAvIGQAAICxCBkAAGAsQgYA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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.barplot(data=credit_card_raw,x='GENDER',y='EMAIL_ID',hue='label')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "KxRnHYc9QHAR",
        "outputId": "0a753126-923e-456f-aca6-368aa0e115e2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: xlabel='GENDER', ylabel='EMAIL_ID'>"
            ]
          },
          "metadata": {},
          "execution_count": 197
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(30,8))\n",
        "sns.countplot(x=credit_card_raw['Type_Occupation'])\n",
        "plt.title(\"Number of people Occupation wise\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 397
        },
        "id": "zApGA6RnzII8",
        "outputId": "c33d27f4-2eac-42dc-adef-41c937375b34"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Number of people Occupation wise')"
            ]
          },
          "metadata": {},
          "execution_count": 177
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 3000x800 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAACWAAAAK9CAYAAACg6ldlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACbHklEQVR4nOzde5hWdb3//9dwGhGcQRRmQBHxkIhingomQU1JJMzcgYe2B1TSItSteCDKU2RipmFairnzkIdfan211DzgEVPCw96WhyQsDRNnIJUZpRhO9++PLu7dCCgu0RF5PK7rvi7vtT73Wu91z80/Xc/WqiiVSqUAAAAAAAAAAADwnrVp7QEAAAAAAAAAAADWVgIsAAAAAAAAAACAggRYAAAAAAAAAAAABQmwAAAAAAAAAAAAChJgAQAAAAAAAAAAFCTAAgAAAAAAAAAAKEiABQAAAAAAAAAAUJAACwAAAAAAAAAAoCABFgAAAAAAAAAAQEECLAAAAIAPyYMPPpiKior84he/aO1RVktDQ0NGjhyZjTbaKBUVFbnoootae6T35cgjj8zmm2/e2mN8rK1t3/HZZ5+dioqK1h4DAAAAWMsJsAAAAICPlauvvjoVFRVZb7318sorr6ywf88998z222/fCpOtfU466aTcfffdmTBhQq699trsu+++rT3SR9IjjzyS//iP/0hNTU0qKyuz+eab56tf/Wpmz57d2qN9IObMmZOzzz47Tz31VGuPAgAAAPCRIMACAAAAPpaam5tz3nnntfYYa7X7778/X/ziF3PKKafksMMOS9++fVt7pI+cSy65JIMHD87TTz+d448/PpdeemlGjhyZG2+8MTvssEMeffTR1h5xjZszZ06+/e1vrzTAuuKKKzJz5swPf6iCTj/99Pzzn/9s7TEAAACAtVy71h4AAAAA4IOw44475oorrsiECRPSs2fP1h7nQ7VgwYJ06tTpfR9n7ty56dKly/sf6GPqkUceyYknnphBgwblrrvuyvrrr1/eN2bMmOy2224ZOXJknn322Wy44YatOOmHp3379q09wnvSrl27tGvnfyIFAAAA3h93wAIAAAA+lr75zW9m6dKl73oXrJdeeikVFRW5+uqrV9hXUVGRs88+u/z+7LPPTkVFRf70pz/lsMMOS3V1dbp165YzzjgjpVIpL7/8cr74xS+mqqoqtbW1ufDCC1d6zqVLl+ab3/xmamtr06lTp+y///55+eWXV1g3Y8aM7Lvvvqmurs7666+fPfbYI4888kiLNctneu655/Kf//mf2XDDDTNo0KB3vOa//OUvOfDAA9O1a9esv/76GThwYO64447y/uWPcSyVSvnxj3+cioqKVFRUvOt3eMEFF2Ty5Mnp3bt3OnbsmD322CPPPPPMCuuff/75jBw5Ml27ds16662XXXfdNb/+9a/f85xJ8uCDD6aioiI33njjan2nb7ds2bJcdNFF2W677bLeeuulpqYmX/3qV/PGG2+862e/853vpKKiItdcc02L+CpJttxyy5x//vl59dVXc/nll69w/QcddFC6deuWjh07Zptttsm3vvWtFmteeeWVjB49Oj179kxlZWX69OmTMWPGZNGiRUn+7+/+dsv/di+99FJ52+abb5799tsv99xzT3bcccest9566devX/7f//t/LT77+uuv55RTTkn//v3TuXPnVFVVZdiwYfn9739fXvPggw/mU5/6VJLkqKOOKv82lv/7OfLII7P55pu3OO6CBQty8sknp1evXqmsrMw222yTCy64IKVSqcW6ioqKHHfccbn11luz/fbbp7KyMtttt13uuuuuVfwF/qVUKmXjjTfOuHHjytuWLVuWLl26pG3btpk/f355+/e+9720a9cub7311iq/x6lTp2bQoEHp0qVLOnfunG222Sbf/OY3W6xpbm7OWWedla222iqVlZXp1atXTjvttDQ3N7/jrAAAAMDHkwALAAAA+Fjq06dPjjjiiFxxxRWZM2fOGj32wQcfnGXLluW8887LgAEDcs455+Siiy7K5z73uWyyySb53ve+l6222iqnnHJKpk2btsLnv/vd7+aOO+7I+PHjc8IJJ2Tq1KkZMmRIi0eh3X///dl9993T1NSUs846K+eee27mz5+fvfbaK4899tgKxzzwwAPzj3/8I+eee26OOeaYVc7e0NCQz3zmM7n77rvz9a9/Pd/97nezcOHC7L///rnllluSJLvvvnuuvfbaJMnnPve5XHvtteX37+RnP/tZLr744owdOzYTJkzIM888k7322isNDQ3lNc8++2wGDhyYP/7xj/nGN76RCy+8MJ06dcoBBxxQPv/qzvlev9OV+epXv5pTTz01u+22W374wx/mqKOOyvXXX5+hQ4dm8eLFq/zcP/7xj9x3330ZPHhw+vTps9I1Bx98cCorK3P77beXt/3hD3/IgAEDcv/99+eYY47JD3/4wxxwwAG57bbbymvmzJmTT3/60/n5z3+egw8+OBdffHEOP/zwPPTQQ/nHP/7xjtezKrNmzcrBBx+cYcOGZdKkSWnXrl0OPPDATJ06tbzmL3/5S2699dbst99++cEPfpBTTz01Tz/9dPbYY4/yv6Ftt902EydOTJIce+yx5d/G7rvvvtLzlkql7L///pk8eXL23Xff/OAHP8g222yTU089tUUwtdxvf/vbfP3rX88hhxyS888/PwsXLsyIESPy2muvrfLaKioqsttuu7X4t/aHP/whjY2NSdIiWnz44Yez0047pXPnzis91rPPPpv99tsvzc3NmThxYi688MLsv//+LY6xbNmy7L///rngggvyhS98IZdcckkOOOCATJ48OQcffPAq5wQAAAA+xkoAAAAAHyNXXXVVKUnp8ccfL/35z38utWvXrnTCCSeU9++xxx6l7bbbrvz+xRdfLCUpXXXVVSscK0nprLPOKr8/66yzSklKxx57bHnbkiVLSptuummpoqKidN5555W3v/HGG6WOHTuWRo0aVd72wAMPlJKUNtlkk1JTU1N5+0033VRKUvrhD39YKpVKpWXLlpW23nrr0tChQ0vLli0rr/vHP/5R6tOnT+lzn/vcCjN9+ctfXq3v58QTTywlKT388MPlbW+++WapT58+pc0337y0dOnSFtc/duzYdz3m8u+wY8eOpb/97W/l7TNmzCglKZ100knlbXvvvXepf//+pYULF5a3LVu2rPSZz3ymtPXWW7/nOVf3Oy2VSqVRo0aVevfuXX7/8MMPl5KUrr/++hbXc9ddd610+7976qmnSklK//Vf//WO380OO+xQ6tq1a/n97rvvXtpggw1Kf/3rX1us+/e/8xFHHFFq06ZN6fHHH1/heMvXLf+7v93y3/+LL75Y3ta7d+9SktIvf/nL8rbGxsZSjx49SjvttFN528KFC1v8/Uulf/1tKysrSxMnTixve/zxx1f5b+bt3/Gtt95aSlI655xzWqwbOXJkqaKiovTCCy+UtyUpdejQocW23//+96UkpUsuuWSFc/2773//+6W2bduWfwMXX3xxqXfv3qVPf/rTpfHjx5dKpVJp6dKlpS5durT4Pb79e5w8eXIpSWnevHmrPNe1115batOmTYvfZqlUKk2ZMqWUpPTII4+846wAAADAx487YAEAAAAfW1tssUUOP/zw/OQnP8mrr766xo77la98pfzfbdu2za677ppSqZTRo0eXt3fp0iXbbLNN/vKXv6zw+SOOOCIbbLBB+f3IkSPTo0eP/OY3v0mSPPXUU5k1a1b+8z//M6+99lr+/ve/5+9//3sWLFiQvffeO9OmTcuyZctaHPNrX/vaas3+m9/8Jp/+9KdbPKawc+fOOfbYY/PSSy/lueeeW70vYSUOOOCAbLLJJuX3n/70pzNgwIDydb3++uu5//77c9BBB+XNN98sX9drr72WoUOHZtasWXnllVcKzflu3+nK3Hzzzamurs7nPve58ix///vfs8suu6Rz58554IEHVvnZN998M0lanHNlNthggzQ1NSVJ5s2bl2nTpuXoo4/OZptt1mLd8sfgLVu2LLfeemu+8IUvZNddd13heO/0KMh30rNnz/zHf/xH+X1VVVWOOOKI/O///m/q6+uTJJWVlWnT5l//c+HSpUvz2muvlR/B9z//8z+Fzvub3/wmbdu2zQknnNBi+8knn5xSqZQ777yzxfYhQ4Zkyy23LL/fYYcdUlVVtdJ/R/9u8ODBWbp0aR599NEk/7rT1eDBgzN48OA8/PDDSZJnnnkm8+fPz+DBg1d5nC5duiRJfvWrX63wb2y5m2++Odtuu2369u3b4nez1157Jck7/m4AAACAjycBFgAAAPCxdvrpp2fJkiU577zz1tgx3x7PVFdXZ7311svGG2+8wvY33nhjhc9vvfXWLd5XVFRkq622yksvvZTkX4+LS5JRo0alW7duLV7//d//nebm5vLj1ZZb1WPw3u6vf/1rttlmmxW2b7vttuX9Rb39upLkE5/4RPm6XnjhhZRKpZxxxhkrXNdZZ52VJJk7d26hOd/tO12ZWbNmpbGxMd27d19hnrfeeqs8y8osD6+Wh1ir8uabb5bXLo+Itt9++1WunzdvXpqamt5xTRFbbbXVCvHWJz7xiSQpf0fLli3L5MmTs/XWW6eysjIbb7xxunXr1uJxfu/VX//61/Ts2XOFUG1Vf8e3/9tKkg033HCl/47+3c4775z111+/HFstD7B23333PPHEE1m4cGF5379HfW938MEHZ7fddstXvvKV1NTU5JBDDslNN93UIsaaNWtWnn322RV+M8u/z3f63QAAAAAfT+1aewAAAACAD9IWW2yRww47LD/5yU/yjW98Y4X9q7qj0NKlS1d5zLZt267WtiQplUqrOen/WR57fP/738+OO+640jWdO3du8b5jx47v+TwftuXXdcopp2To0KErXbPVVlt9qPN07949119//Ur3d+vWbZWf3WqrrdKuXbv84Q9/WOWa5ubmzJw5c6V3snq/ivxu3825556bM844I0cffXS+853vpGvXrmnTpk1OPPHEVd4Nak0r+u+offv2GTBgQKZNm5YXXngh9fX1GTx4cGpqarJ48eLMmDEjDz/8cPr27fuOf9eOHTtm2rRpeeCBB3LHHXfkrrvuyo033pi99tor99xzT9q2bZtly5alf//++cEPfrDSY/Tq1Wv1LxgAAAD4WBBgAQAAAB97p59+eq677rp873vfW2HfhhtumCSZP39+i+3v505Q72b5Ha6WK5VKeeGFF7LDDjskSfkRbFVVVRkyZMgaPXfv3r0zc+bMFbY///zz5f1Fvf26kuRPf/pTNt988yT/iuGSf8Uy73Zd73XOd/tOV2bLLbfMvffem9122+09B2ydOnXKZz/72dx///3561//utLv7aabbkpzc3P222+/JP93/c8888wqj9utW7dUVVW945qk5e92+WPzklX/bpfffezfw60//elPSVL++/ziF7/IZz/72fz0pz9t8dn58+e3uLvbe3kMYu/evXPvvfe2uBNYsmZ+b283ePDgfO9738u9996bjTfeOH379k1FRUW22267PPzww3n44YfLf4t30qZNm+y9997Ze++984Mf/CDnnntuvvWtb+WBBx4oPyLx97//ffbee+/Cj4QEAAAAPl48ghAAAAD42Ntyyy1z2GGH5fLLL099fX2LfVVVVdl4440zbdq0FtsvvfTSD2yen/3sZy0eXfeLX/wir776aoYNG5Yk2WWXXbLlllvmggsuyFtvvbXC5+fNm1f43J///Ofz2GOPZfr06eVtCxYsyE9+8pNsvvnm6devX+Fj33rrrXnllVfK7x977LHMmDGjfF3du3fPnnvumcsvvzyvvvrqCp//9+t6r3O+23e6MgcddFCWLl2a73znOyvsW7JkyQpR3tudfvrpKZVKOfLII/PPf/6zxb4XX3wxp512Wnr06JGvfvWrSf4VV+2+++658sorM3v27Bbrl9/hqU2bNjnggANy22235YknnljhnMvXLY/0/v13u2DBglxzzTUrnXXOnDm55ZZbyu+bmprys5/9LDvuuGNqa2uT/OvuU2+/09TNN9/c4m+a/Cs+S1aMFlfm85//fJYuXZof/ehHLbZPnjw5FRUV7/j3ea8GDx6c5ubmXHTRRRk0aFA5jho8eHCuvfbazJkzJ4MHD37HY7z++usrbFt+F7rm5uYk//rdvPLKK7niiitWWPvPf/4zCxYseJ9XAgAAAKxt3AELAAAAWCd861vfyrXXXpuZM2dmu+22a7HvK1/5Ss4777x85Stfya677ppp06aV7w70QejatWsGDRqUo446Kg0NDbnooouy1VZb5Zhjjknyrwjnv//7vzNs2LBst912Oeqoo7LJJpvklVdeyQMPPJCqqqrcdttthc79jW98I//f//f/ZdiwYTnhhBPStWvXXHPNNXnxxRfzy1/+Mm3aFP//62211VYZNGhQxowZUw5hNtpoo5x22mnlNT/+8Y8zaNCg9O/fP8ccc0y22GKLNDQ0ZPr06fnb3/6W3//+94XmfLfvdGX22GOPfPWrX82kSZPy1FNPZZ999kn79u0za9as3HzzzfnhD3+YkSNHrvLzu+++ey644IKMGzcuO+ywQ4488sj06NEjzz//fK644oosW7Ysv/nNb8p3q0qSiy++OIMGDcrOO++cY489Nn369MlLL72UO+64I0899VSSfz0K8J577skee+yRY489Nttuu21effXV3Hzzzfntb3+bLl26ZJ999slmm22W0aNH59RTT03btm1z5ZVXplu3bivEXUnyiU98IqNHj87jjz+empqaXHnllWloaMhVV11VXrPffvtl4sSJOeqoo/KZz3wmTz/9dK6//vrynbuW23LLLdOlS5dMmTIlG2ywQTp16pQBAwakT58+K5z3C1/4Qj772c/mW9/6Vl566aV88pOfzD333JNf/epXOfHEE8sh2ZpQV1eXdu3aZebMmTn22GPL23ffffdcdtllSfKuAdbEiRMzbdq0DB8+PL17987cuXNz6aWXZtNNN82gQYOSJIcffnhuuummfO1rX8sDDzyQ3XbbLUuXLs3zzz+fm266KXffffcH8thJAAAA4KNLgAUAAACsE7baaqscdthhK71D0Jlnnpl58+blF7/4RW666aYMGzYsd955Z7p37/6BzPLNb34zf/jDHzJp0qS8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yCH54x//mH79+uXxxx/PrrvumiS566678vnPfz5/+9vf0rNnz3edo6mpKdXV1WlsbExVVdVqzz/vsutWey0frm5jDmvtEQAAAAAAAAAAWIutblPUqnfA+vWvf51dd901Bx54YLp3756ddtopV1xxRXn/iy++mPr6+gwZMqS8rbq6OgMGDMj06dOTJNOnT0+XLl3K8VWSDBkyJG3atMmMGTNWet7m5uY0NTW1eAEAAAAAAAAAALxXrRpg/eUvf8lll12WrbfeOnfffXfGjBmTE044Iddcc02SpL6+PklSU1PT4nM1NTXlffX19enevXuL/e3atUvXrl3La95u0qRJqa6uLr969eq1pi8NAAAAAAAAAABYB7RqgLVs2bLsvPPOOffcc7PTTjvl2GOPzTHHHJMpU6Z8oOedMGFCGhsby6+XX375Az0fAAAAAAAAAADw8dSqAVaPHj3Sr1+/Ftu23XbbzJ49O0lSW1ubJGloaGixpqGhobyvtrY2c+fObbF/yZIlef3118tr3q6ysjJVVVUtXgAAAAAAAAAAAO9VqwZYu+22W2bOnNli25/+9Kf07t07SdKnT5/U1tbmvvvuK+9vamrKjBkzUldXlySpq6vL/Pnz8+STT5bX3H///Vm2bFkGDBjwIVwFAAAAAAAAAACwrmrXmic/6aST8pnPfCbnnntuDjrooDz22GP5yU9+kp/85CdJkoqKipx44ok555xzsvXWW6dPnz4544wz0rNnzxxwwAFJ/nXHrH333bf86MLFixfnuOOOyyGHHJKePXu24tUBAAAAAAAAAAAfd60aYH3qU5/KLbfckgkTJmTixInp06dPLrroohx66KHlNaeddloWLFiQY489NvPnz8+gQYNy1113Zb311iuvuf7663Pcccdl7733Tps2bTJixIhcfPHFrXFJAAAAAAAAAADAOqSiVCqVWnuI1tbU1JTq6uo0NjamqqpqtT8377LrPsCpeD+6jTmstUcAAAAAAAAAAGAttrpNUZsPcSYAAAAAAAAAAICPFQEWAAAAAAAAAABAQQIsAAAAAAAAAACAggRYAAAAAAAAAAAABQmwAAAAAAAAAAAAChJgAQAAAAAAAAAAFCTAAgAAAAAAAAAAKEiABQAAAAAAAAAAUJAACwAAAAAAAAAAoCABFgAAAAAAAAAAQEECLAAAAAAAAAAAgIIEWAAAAAAAAAAAAAUJsAAAAAAAAAAAAAoSYAEAAAAAAAAAABQkwAIAAAAAAAAAAChIgAUAAAAAAAAAAFCQAAsAAAAAAAAAAKAgARYAAAAAAAAAAEBBAiwAAAAAAAAAAICCBFgAAAAAAAAAAAAFCbAAAAAAAAAAAAAKEmABAAAAAAAAAAAUJMACAAAAAAAAAAAoSIAFAAAAAAAAAABQkAALAAAAAAAAAACgIAEWAAAAAAAAAABAQQIsAAAAAAAAAACAggRYAAAAAAAAAAAABQmwAAAAAAAAAAAAChJgAQAAAAAAAAAAFCTAAgAAAAAAAAAAKEiABQAAAAAAAAAAUJAACwAAAAAAAAAAoCABFgAAAAAAAAAAQEECLAAAAAAAAAAAgIIEWAAAAAAAAAAAAAUJsAAAAAAAAAAAAAoSYAEAAAAAAAAAABQkwAIAAAAAAAAAAChIgAUAAAAAAAAAAFCQAAsAAAAAAAAAAKAgARYAAAAAAAAAAEBBAiwAAAAAAAAAAICCBFgAAAAAAAAAAAAFCbAAAAAAAAAAAAAKEmABAAAAAAAAAAAUJMACAAAAAAAAAAAoSIAFAAAAAAAAAABQkAALAAAAAAAAAACgIAEWAAAAAAAAAABAQQIsAAAAAAAAAACAggRYAAAAAAAAAAAABQmwAAAAAAAAAAAAChJgAQAAAAAAAAAAFCTAAgAAAAAAAAAAKEiABQAAAAAAAAAAUJAACwAAAAAAAAAAoCABFgAAAAAAAAAAQEECLAAAAAAAAAAAgIIEWAAAAAAAAAAAAAUJsAAAAAAAAAAAAAoSYAEAAAAAAAAAABQkwAIAAAAAAAAAAChIgAUAAAAAAAAAAFCQAAsAAAAAAAAAAKAgARYAAAAAAAAAAEBBAiwAAAAAAAAAAICCWjXAOvvss1NRUdHi1bdv3/L+hQsXZuzYsdloo43SuXPnjBgxIg0NDS2OMXv27AwfPjzrr79+unfvnlNPPTVLliz5sC8FAAAAAAAAAABYB7Vr7QG222673HvvveX37dr930gnnXRS7rjjjtx8882prq7Occcdly996Ut55JFHkiRLly7N8OHDU1tbm0cffTSvvvpqjjjiiLRv3z7nnnvuh34tAAAAAAAAAADAuqXVA6x27dqltrZ2he2NjY356U9/mhtuuCF77bVXkuSqq67Ktttum9/97ncZOHBg7rnnnjz33HO59957U1NTkx133DHf+c53Mn78+Jx99tnp0KHDh305AAAAAAAAAADAOqRVH0GYJLNmzUrPnj2zxRZb5NBDD83s2bOTJE8++WQWL16cIUOGlNf27ds3m222WaZPn54kmT59evr375+amprymqFDh6apqSnPPvvsKs/Z3NycpqamFi8AAAAAAAAAAID3qlUDrAEDBuTqq6/OXXfdlcsuuywvvvhiBg8enDfffDP19fXp0KFDunTp0uIzNTU1qa+vT5LU19e3iK+W71++b1UmTZqU6urq8qtXr15r9sIAAAAAAAAAAIB1Qqs+gnDYsGHl/95hhx0yYMCA9O7dOzfddFM6duz4gZ13woQJGTduXPl9U1OTCAsAAAAAAAAAAHjPWv0RhP+uS5cu+cQnPpEXXnghtbW1WbRoUebPn99iTUNDQ2pra5MktbW1aWhoWGH/8n2rUllZmaqqqhYvAAAAAAAAAACA9+ojFWC99dZb+fOf/5wePXpkl112Sfv27XPfffeV98+cOTOzZ89OXV1dkqSuri5PP/105s6dW14zderUVFVVpV+/fh/6/AAAAAAAAAAAwLqlVR9BeMopp+QLX/hCevfunTlz5uSss85K27Zt8+UvfznV1dUZPXp0xo0bl65du6aqqirHH3986urqMnDgwCTJPvvsk379+uXwww/P+eefn/r6+px++ukZO3ZsKisrW/PSAAAAAAAAAACAdUCrBlh/+9vf8uUvfzmvvfZaunXrlkGDBuV3v/tdunXrliSZPHly2rRpkxEjRqS5uTlDhw7NpZdeWv5827Ztc/vtt2fMmDGpq6tLp06dMmrUqEycOLG1LgkAAAAAAAAAAFiHVJRKpVJrD9HampqaUl1dncbGxlRVVa325+Zddt0HOBXvR7cxh7X2CAAAAAAAAAAArMVWtylq8yHOBAAAAAAAAAAA8LEiwAIAAAAAAAAAAChIgAUAAAAAAAAAAFCQAAsAAAAAAAAAAKAgARYAAAAAAAAAAEBBAiwAAAAAAAAAAICCBFgAAAAAAAAAAAAFCbAAAAAAAAAAAAAKEmABAAAAAAAAAAAUJMACAAAAAAAAAAAoSIAFAAAAAAAAAABQkAALAAAAAAAAAACgIAEWAAAAAAAAAABAQQIsAAAAAAAAAACAggRYAAAAAAAAAAAABQmwAAAAAAAAAAAAChJgAQAAAAAAAAAAFCTAAgAAAAAAAAAAKEiABQAAAAAAAAAAUJAACwAAAAAAAAAAoCABFgAAAAAAAAAAQEECLAAAAAAAAAAAgIIEWAAAAAAAAAAAAAUJsAAAAAAAAAAAAAoSYAEAAAAAAAAAABQkwAIAAAAAAAAAAChIgAUAAAAAAAAAAFCQAAsAAAAAAAAAAKAgARYAAAAAAAAAAEBBAiwAAAAAAAAAAICCBFgAAAAAAAAAAAAFCbAAAAAAAAAAAAAKEmABAAAAAAAAAAAUJMACAAAAAAAAAAAoSIAFAAAAAAAAAABQkAALAAAAAAAAAACgIAEWAAAAAAAAAABAQQIsAAAAAAAAAACAggRYAAAAAAAAAAAABQmwAAAAAAAAAAAAChJgAQAAAAAAAAAAFCTAAgAAAAAAAAAAKEiABQAAAAAAAAAAUJAACwAAAAAAAAAAoCABFgAAAAAAAAAAQEECLAAAAAAAAAAAgIIEWAAAAAAAAAAAAAUJsAAAAAAAAAAAAAoSYAEAAAAAAAAAABQkwAIAAAAAAAAAAChIgAUAAAAAAAAAAFCQAAsAAAAAAAAAAKAgARYAAAAAAAAAAEBBAiwAAAAAAAAAAICCBFgAAAAAAAAAAAAFCbAAAAAAAAAAAAAKKhRg7bXXXpk/f/4K25uamrLXXnu935kAAAAAAAAAAADWCoUCrAcffDCLFi1aYfvChQvz8MMPv++hAAAAAAAAAAAA1gbt3sviP/zhD+X/fu6551JfX19+v3Tp0tx1113ZZJNN1tx0AAAAAAAAAAAAH2HvKcDacccdU1FRkYqKipU+arBjx4655JJL1thwAAAAAAAAAAAAH2XvKcB68cUXUyqVssUWW+Sxxx5Lt27dyvs6dOiQ7t27p23btmt8SAAAAAAAAAAAgI+i9xRg9e7dO0mybNmyD2QYAAAAAAAAAACAtcl7CrD+3axZs/LAAw9k7ty5KwRZZ5555vseDAAAAAAAAAAA4KOuUIB1xRVXZMyYMdl4441TW1ubioqK8r6KigoBFgAAAAAAAAAAsE4oFGCdc845+e53v5vx48ev6XkAAAAAAAAAAADWGm2KfOiNN97IgQceuKZnAQAAAAAAAAAAWKsUCrAOPPDA3HPPPWt0kPPOOy8VFRU58cQTy9sWLlyYsWPHZqONNkrnzp0zYsSINDQ0tPjc7NmzM3z48Ky//vrp3r17Tj311CxZsmSNzgYAAAAAAAAAALAyhR5BuNVWW+WMM87I7373u/Tv3z/t27dvsf+EE054T8d7/PHHc/nll2eHHXZosf2kk07KHXfckZtvvjnV1dU57rjj8qUvfSmPPPJIkmTp0qUZPnx4amtr8+ijj+bVV1/NEUcckfbt2+fcc88tcmkAAAAAAAAAAACrraJUKpXe64f69Omz6gNWVOQvf/nLah/rrbfeys4775xLL70055xzTnbcccdcdNFFaWxsTLdu3XLDDTdk5MiRSZLnn38+2267baZPn56BAwfmzjvvzH777Zc5c+akpqYmSTJlypSMHz8+8+bNS4cOHVZrhqamplRXV6exsTFVVVWrPfu8y65b7bV8uLqNOay1RwAAAAAAAAAAYC22uk1RoUcQvvjii6t8vZf4KknGjh2b4cOHZ8iQIS22P/nkk1m8eHGL7X379s1mm22W6dOnJ0mmT5+e/v37l+OrJBk6dGiampry7LPPrvKczc3NaWpqavECAAAAAAAAAAB4rwo9gnBN+fnPf57/+Z//yeOPP77Cvvr6+nTo0CFdunRpsb2mpib19fXlNf8eXy3fv3zfqkyaNCnf/va33+f0AAAAAAAAAADAuq5QgHX00Ue/4/4rr7zyXY/x8ssv57/+678yderUrLfeekXGKGzChAkZN25c+X1TU1N69er1oc4AAAAAAAAAAACs/QoFWG+88UaL94sXL84zzzyT+fPnZ6+99lqtYzz55JOZO3dudt555/K2pUuXZtq0afnRj36Uu+++O4sWLcr8+fNb3AWroaEhtbW1SZLa2to89thjLY7b0NBQ3rcqlZWVqaysXK05AQAAAAAAAAAAVqVQgHXLLbessG3ZsmUZM2ZMttxyy9U6xt57752nn366xbajjjoqffv2zfjx49OrV6+0b98+9913X0aMGJEkmTlzZmbPnp26urokSV1dXb773e9m7ty56d69e5Jk6tSpqaqqSr9+/YpcGgAAAAAAAAAAwGorFGCtTJs2bTJu3LjsueeeOe200951/QYbbJDtt9++xbZOnTplo402Km8fPXp0xo0bl65du6aqqirHH3986urqMnDgwCTJPvvsk379+uXwww/P+eefn/r6+px++ukZO3asO1wBAAAAAAAAAAAfuDUWYCXJn//85yxZsmSNHW/y5Mlp06ZNRowYkebm5gwd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5sxx2uacOXMc9UirVq0sNjbWJDml/UppaWk2bNgwCw8PNz8/P6tQoYL169fPzDJf//Tbzx9//GGPP/64VaxY0QICAqxu3br21VdfObYZExOTKb+5mlZQ/fXXXybJli1b5nY5T+v+K7mrr91dK1c2b95srVq1sqCgIAsODrZGjRrZunXrbOnSpZmua/r9YvLkyda4cWMLCgqy8uXL2xNPPGGHDx82M7OkpKRM602cONHltMLm1KlTFhQUZDt37rQuXbrYW2+95TR/7ty51qRJE/P397cyZcrYgw8+6Jh37tw5e+2116xSpUrm5+dn1apVs3//+9+O+cuWLbPbbrvN/Pz8LCwszAYNGuR0j85YD5iZNWjQwHHNzNzXDa6uW0xMjJmZ/fjjj9a8eXMLDQ210qVLW4cOHWz37t2O7aavO3PmTGvVqpUFBARY/fr1bdWqVWZmbvPS+PHjHXXRTTfdZI888ojX5z87WbXB0tOXXv+5KnMjRoywcuXKWVBQkPXq1csGDRpkDRo0yLTtUaNGWVhYmJUuXdpeeOEFu3DhQpbpyarsuUrDkSNHrHHjxvbggw/auXPnsk1zepvBFXfX+tKlS/b2229bZGSkFS9e3OrXr28zZsxwWv+3336zDh06WHBwsAUFBdkdd9zhyA+5eR4Kax2Ufo7q1q1rX375pWP61KlTrX79+tapU6dcK3tm2d+LzcxOnjxpXbt2tRIlSlhYWJiNHj3aWrZsaf3793csc+7cOXvllVesYsWKVqJECWvatKktXbrUMT89D86ZM8eioqLM19fXkpKSbOnSpXbbbbdZiRIlLDQ01KKjo23v3r25e1LziKdtwYztuP3791vnzp0tNDTUSpUqZQ888IAlJSU55q9du9batm1rZcqUsZCQEGvRooVt2LDBafvZtfWyqiPmz59vtWvXtsDAQGvfvr0dPHjQsU5qaqr169fPkd9ee+01e/rpp7Psu5qZ7d271+6//34rWbKklShRwurUqWPff//9Vd1TXJX7rOqCwi6r+9bixYsdecDMdT/9yjxw4sQJK168uP3www9O25k1a5YFBQXZ6dOnzSz7vJmenpEjR1qFChUsMjLSzK7vvfxacjdWY+a+Lfbdd99ZaGioXbx40czMNm3aZJJs0KBBjvV79epl3bp1y3L7edUuMPv/95NNmza5nJ/deIDZ5bbom2++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srtpea1r16529913Oz7fdtttNmPGDOvdu7cNHTrUzC4/jPX393eUQXfB/mbOZc7dlwauVQD7vHnzrGbNmo7P2QVAmnn+pd8r03L69GmLiIiw0qVL26effmq7du2y559/3kJCQuyee+6xb775xuLj4+3BBx+0qKgoR72U3cO99DSXLl3axo8fbwkJCfbOO+9YkSJFHA/rMiooD+GzGwPypE5zN3Zg5nkAVsa3JHh6b6hTp47jHJldHvu/8tplRHvwspz08zzte2d8YDpo0CC39/vs8k5Ox49HjBjhVH+amR04cMARkOFKQckPV95b09LSbOHChebv728DBw50zM/Y7jVzPofp435X/pLFE0884Xb8I6s+fcY2WLVq1TK9XWbEiBHWrFkzl9vNrt2T3fayqjey60NkTH92Y0a5fVyePmsdNWqUNW7c2PE5/e0vVwbYupqWH3g6lplevtLnpwcRP/DAA263X1DKbGHkrs2QG+2FiIgIpzdimV2+zzz33HNO01asWGFFihRxvBnW2+d8LVu2dHrDU7qcjIkVlnHEK8utv7+/SbIiRYrYf/7zHzPzrB3gSsb+WMY60FW/7vfff3cKlrtw4YKVLVs2y7EHVzKOS/74449WtGhRpy/cZHwDlqdjTNk9R3CVH93VW9dLEaFAWbRokdq0aaObb75ZwcHBeuqpp3Ts2DGdOXPGsUzRokV12223OT7Xrl1bJUuWVFxcnCQpLi5OzZs3d9pu8+bNHfPTNWnSxOnzli1b9OabbyooKMjx9+yzzyo5Odlp/xnXe/755/X111+rYcOGeu2117Rq1aqrOwmFjJl5tFxcXJzCw8MVHh7umFanTh2naytJERERKleunOPzli1blJKSojJlyjhdu6SkJCUmJrrcV/fu3bV582bVqlVLL774on766ads05eSkqKBAwcqKipKJUuWVFBQkOLi4rR//36Pjq+wyc3r6mmZfemll7RmzRr9/PPPuvnmm7Pdd/369R3/BwYGKiQkREeOHJF0+bfAr6xHJKlp06Zut9e5c2edPXtWVatW1bPPPqvZs2fr4sWLbte5dOmSRowYoXr16ql06dIKCgrSggULbqh888gjj+jgwYOaO3eu7rnnHi1btkyNGjVSbGysYxlP6v4rZVdf5/Ravfzyy3rmmWfUtm1bvfvuu1nWHVfasGGDOnbsqMqVKys4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mzZNw4YNU1xcnLZt26Z//vOfki6f55iYGPXs2VPffvutkpKStGzZMsdPI7zwwgs6cOCA+vXrp507d2rOnDkaNmyYXn75ZUfQ3V133aUpU6ZoxYoV2rZtm2JiYnL0UwnS5Z9u8fHx0bx583T06FGlpKSoVKlSKlOmjD777DPt3r1bS5Ys0csvv5zjc+QqL82bN0/jxo3T5s2btW/fPk2ePFlpaWlOP32WH/Tr109ffPGFJk2apISEBI0cOVJbt269qp/+ya7spfP19dXUqVPVoEED3XXXXTp06NDVHo4k19c6ODhYAwcO1EsvvaRJkyYpMTFRGzdu1IcffqhJkyZJkvr27auTJ0/q8ccf1/r165WQkKApU6YoPj7eq3Tc6HWQr6+v4uLitGPHjkzlNbfKXnb34uDgYMXExOjVV1/V0qVLtX37dvXq1UtFihRx5PGaNWuqW7duevrppzVr1iwlJSVp7dq1euedd/T9999nue+kpCQNHjxYq1ev1r59+/TTTz8pISGh0PQVPGkLZtStWzeVLVtWnTp10ooVKxz1/Ysvvuj4GbMaNWpoypQpiouL05o1a9StW7dceVNidvr166d33nlHc+bMUXx8vPr376+//vrLbV03YMAALViwQElJSdq4caOWLl3quL7X8p5yIzl//rwOHTqk33//XRs3btTbb7+tTp066f7779fTTz+do221aNFCYWFh6tatm6pUqeL0szae5E1XCsq93FOuxmoOHDjgUVusVKlSql+/vqZOneoItmrRooU2btyoXbt2ZflT01LetwvSxcfHZzr+1NRUj8cDVq5cqffee0+7du3S+PHjNWPGDPXv31/S9R3zkKQ1a9bo7bff1vr167V//37NmjVLR48eLTT3oJzIrpxGRkZqzZo12rt3r/744w+lpaWpRo0aWr9+vRYsWKBdu3bpf//3f3PtCwh5oXXr1i6DIVu0aKEff/xRa9eudfz8oDfB/q4eDFyrAPY5c+aoQ4cOmdqO7gIgvfni19Xw5OGeN/L7Q3hPxoA8qdPcjR1Il/v+H330kTZt2qT169erd+/ebn8iPiuRkZFKSkrS5s2b9ccff+j8+fOOec8884wmTZqkuLg4xcTEuN0O7cHM3PXzJHnU9+7du7cSEhL06quvKj4+Xl999ZVj7DYr2eWdnOrTp4/+/PNPPfHEE1q3bp0SExO1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iRMnOk2fOHGiHn30UQUHBzumvfrqq3rllVe0adMmNWvWTB07dnT8PODx48d111136dZbb9X69es1f/58HT58WI899li2afjzzz+1YMECvfDCCwoICHCaFxYWpm7dumn69OkyM0nS008/rWnTpmncuHGKi4vTv/71L8cbvH7//Xe1aNFC/v7+WrJkiTZs2KCePXvq4sWLOTovo0aNUoMGDbRp0ya9/vrr6t+/vxYuXOiYX6RIEY0bN07bt2/XpEmTtGTJEr322muSpOjoaI0dO1YhISFKTk5WcnKyBg4cmGkfaWlp6tSpk/78808tX75cCxcu1J49e9SlSxen5RITE/Xtt99q3rx5mjdvnpYvX6533303R8cDAAAAAACArPEThAAAAAAAALgqzzzzjKKjo5WcnKwKFSroyJEj+uGHH7Ro0SKn5fr27atHHnlEkvTJJ59o/vz5+uKLL/Taa6/po48+0q233qq3337bsfyECRMUHh6uXbt2qWbNmlnuPyEhQWamqKgol/OjoqL0119/6ejRozp+/Li++eYbLVy4UG3btpUkVa1a1bHs+PHjFRoaqq+//lrFihWTJLf7zkrz5s31+uuvO9ZfuXKlxowZo3bt2kmS01u0IiMjNXLkSPXu3Vsff/yx/Pz8nN4ylpXFixdr27ZtSkpKUnh4uCRp8uTJuuWWW7Ru3Trddtttki4HasXGxjqC4Z566iktXrxYb731Vo6PCwAAAAAAAJnxBiwAAAAAAABclaZNm+qWW27RpEmTJElffvmlIiIi1KJFC6flmjVr5vi/aNGiatKkieLi4iRJW7Zs0dKlSxUUFOT4q127tiQ5fv4vO+lvuHJn8+bN8vX1VcuWLbOcf+eddzqCr7x15bGmf04/VklatGiR2rRpo5tvvlnBwcF66qmndOzYMZ05c8bjfcTFxSk8PNwRfCVJderUUcmSJZ32FRkZ6fQmsvQgOQAAAAAAAOQOArAAAAAAAABw1Z555hnFxsZKuvzzgz169JCPj4/H66ekpKhjx47avHmz019CQkKmQK6MqlevLh8fH6egoyvFxcWpVKlSKleuXKafKMwou/lFihTJFOiVmprqdp2M9u7dq/vvv1/169fXzJkztWHDBo0fP16SdOHChRxtyxMZg8l8fHyUlpaW6/sBAAAAAAC4URGABQAAAAAAgKv25JNPat++fRo3bpx27NihmJiYTMv8+uuvjv8vXryoDRs2OH42sFGjRtq+fbsiIyNVvXp1p7/AwEC3+y5TpozatWunjz/+WGfPnnWad+jQIU2dOlVdunSRj4+P6tWrp7S0NC1fvtzlturXr68VK1ZkGVRVrlw5JScnOz5funRJv/32m9tjTf+cfqwbNmxQWlqa3n//fd1+++2qWbOmDh486LS8n5+fLl265Pa4o6KidODAAR04cMAxbceOHTp+/Ljq1Knjdl0AAAAAAADkHgKwAAAAAAAAcNVKlSqlhx9+WK+++qruvvtuVapUKdMy48eP1+zZs7Vz50716dNHf/31l3r27ClJ6tOnj/7880898cQTWrdunRITE7VgwQL16NEj20AkSfroo490/vx5tW/fXj///LMOHDig+fPnq127drr55pv11ltvSbr8c3wxMTHq2bOnvv32WyUlJWnZsmX65ptvJEl9+/bVyZMn9fjjj2v9+vVKSEjQlClTFB8fL0m666679P333+v777/Xzp079fzzz+v48eOZ0rNy5Uq999572rVrl8aPH68ZM2aof//+ki6/sSs1NVUffvih9uzZoylTpujTTz91Wj8yMlIpKSlavHix/vjjD5c/Tdi2bVvVq1dP3bp108aNG7V27Vo9/fTTatmypZo0aZLtOQMAAAAAAEDuIAALAAAAAAAAuaJXr166cOGCI6gqo3fffVfvvvuuGjRooF9++UVz585V2bJlJUkVK1bUypUrdenSJd19992qV6+eBgwYoJIlS6pIkeyHsGrUqKH169eratWqeuyxx1StWjU999xzat26tVavXq3SpUs7lv3kk0/06KOP6oUXXlDt2rX17LPP6vTp05Iuv01ryZIlSklJUcuWLdW4cWN9/vnnjp/x69mzp2JiYhyBTlWrVlXr1q0zpeeVV17R+vXrdeutt2rkyJEaPXq02rdvL0lq0KCBRo8erX/+85+qW7eupk6dqnfeecdp/ejoaPXu3VtdunRRuXLl9N5772Xah4+Pj+bMmaNSpUqpRYsWatu2rapWrarp06dne74AAAAAAACQe3zMzPI6EQAAAAAAACj4pkyZopdeekkHDx6Un5+fY/revXtVpUoVbdq0SQ0bNsy7BF4nkZGRGjBggAYMGJDXSQEAAAAAAMB1UDSvEwAAAAAAAICC7cyZM0pOTta7776rv//9707BVwAAAAAAAEBhx08QAgAAAAAA4Kq89957ql27tsLCwjR48OBc3/7+/fsVFBSU5d/+/ftzfZ8AAAAAAACAp/gJQgAAAAAAAORrFy9e1N69e7OcHxkZqaJFedE7AAAAAAAA8gYBWAAAAAAAAAAAAAAAAADgJX6CEAAAAAAAAAAAAAAAAAC8RAAWAAAAAAAAAAAAAAAAAHiJACwAAAAAAAAAAAAAAAAA8BIBWAAAAAAAAAAAAAAAAADgJQKwAAAAAAAAAAAAAAAAAMBLBGABAAAAAAAAAAAAAAAAgJcIwAIAAAAAAAAAAAAAAAAAL/0/M+7HA6GgZFUAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Numerical variables distribution\n",
        "sns.distplot(credit_card_raw['Annual_income'])\n",
        "plt.title(\"Distribution of Annual Income\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "id": "xuJ9TkR-Q4h0",
        "outputId": "eb1e7343-c975-4824-f218-ff1d5e70dbdb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Distribution of Annual Income')"
            ]
          },
          "metadata": {},
          "execution_count": 178
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAioAAAHHCAYAAACRAnNyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABaVklEQVR4nO3deXgTdf4H8PckadL7vqEHZ0GOUkSQS0BukGNVQBehIKy7itciq4K74g2sgHjww1VZKiggHoCLyiGnIiL3JSBHKaVQCr3PtEm+vz/ShIa20CPJTNr363nyQCbTmU+G0L77vUYSQggQERERKZBK7gKIiIiIqsOgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCjcIrr7wCSZKccq6+ffuib9++1uc7duyAJEn46quvnHL+SZMmITY21innqquCggJMnToV4eHhkCQJzz77rNwlOVVsbCwmTZokdxlELoFBhVxOUlISJEmyPtzd3REZGYnBgwfjvffeQ35+vl3Oc/nyZbzyyis4fPiwXY5nT0qurSbeeustJCUl4fHHH8eKFSswYcKE236N0WhEZGQkJEnCDz/84IQq5SdJEp588km5yyCSlUbuAojq6rXXXkOzZs1QVlaG9PR07NixA88++ywWLlyIb7/9Fh07drTu+89//hMvvvhirY5/+fJlvPrqq4iNjUWnTp1q/HWbN2+u1Xnq4la1ffzxxzCZTA6voT62bduGu+++G7Nnz67V11y5cgWxsbH4/PPPMXToUAdWSERKwaBCLmvo0KHo0qWL9fnMmTOxbds23HfffRg5ciROnjwJDw8PAIBGo4FG49iPe1FRETw9PaHVah16nttxc3OT9fw1kZGRgTvuuKNWX/PZZ5+hc+fOSExMxKxZs1BYWAgvLy8HVUhESsGuH2pQ7r33XvzrX/9CSkoKPvvsM+v2qsaobNmyBb169YK/vz+8vb0RFxeHWbNmATCPK7nrrrsAAJMnT7Z2MyUlJQEwj0Np3749Dhw4gHvuuQeenp7Wr715jIqF0WjErFmzEB4eDi8vL4wcORKpqak2+1Q3dqHiMW9XW1VjVAoLC/Hcc88hKioKOp0OcXFxmD9/Pm6+ebqlq2HdunVo3749dDod2rVrh40bN1Z9wW+SkZGBKVOmICwsDO7u7oiPj8enn35qfd0yXic5ORnfffedtfYLFy7c8rjFxcVYu3YtHnroIYwdOxbFxcVYv359pf0mTZoEb29vpKWlYfTo0fD29kZISAhmzJgBo9Fo3e/ChQuQJAnz58/HRx99hBYtWkCn0+Guu+7Cvn37bI5Z3b9nVdd5/vz56NGjB4KCguDh4YE777zTrmOTLNdvzZo1ePPNN9G0aVO4u7ujf//+OHv2bKX99+7di2HDhiEgIABeXl7o2LEj3n33XZt9tm3bht69e8PLywv+/v4YNWoUTp48abOP5f/PH3/8gUceeQR+fn4ICQnBv/71LwghkJqailGjRsHX1xfh4eFYsGBBpVr0ej1mz56Nli1bQqfTISoqCs8//zz0er3drg81TAwq1OBYxjvcqgvmxIkTuO+++6DX6/Haa69hwYIFGDlyJHbv3g0AaNu2LV577TUAwGOPPYYVK1ZgxYoVuOeee6zHyMzMxNChQ9GpUycsWrQI/fr1u2Vdb775Jr777ju88MILePrpp7FlyxYMGDAAxcXFtXp/NamtIiEERo4ciXfeeQdDhgzBwoULERcXh3/84x+YPn16pf1//vlnPPHEE3jooYfw73//GyUlJXjggQeQmZl5y7qKi4vRt29frFixAuPHj8fbb78NPz8/TJo0yfrDsW3btlixYgWCg4PRqVMna+0hISG3PPa3336LgoICPPTQQwgPD0ffvn3x+eefV7mv0WjE4MGDERQUhPnz56NPnz5YsGABPvroo0r7rly5Em+//Tb++te/4o033sCFCxdw//33o6ys7Jb1VOfdd99FQkICXnvtNbz11lvQaDQYM2YMvvvuuzodrzpz587F2rVrMWPGDMycORO//vorxo8fb7PPli1bcM899+D333/HM888gwULFqBfv37YsGGDdZ8ff/wRgwcPRkZGBl555RVMnz4dv/zyC3r27FlleBw3bhxMJhPmzp2Lbt264Y033sCiRYswcOBANGnSBPPmzUPLli0xY8YM7Nq1y/p1JpMJI0eOxPz58zFixAi8//77GD16NN555x2MGzfOrteGGiBB5GKWLVsmAIh9+/ZVu4+fn59ISEiwPp89e7ao+HF/5513BABx7dq1ao+xb98+AUAsW7as0mt9+vQRAMSHH35Y5Wt9+vSxPt++fbsAIJo0aSLy8vKs29esWSMAiHfffde6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H32EadOmyVkWERERKYSsQeWuu+7C2rVrsWrVKrRv3x6vv/46Fi1ahPHjx8tZFhERESmE7Evo33fffbjvvvvkLoOIiIgUSPYl9ImIiIiqw6BCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaBCREREisWgQkRERIrFoEJERESKxaDSAJ3NyMfne1NgMJrkLoWIiKheNHIXQPZ1vUCPhz/ei2v5elzPL8UzA1rJXRIREVGdsUWlATGZBJ5bcwTX8vUAgMXbz+LctQKZqyIiIqo7BpUG5JOfz2PnH9fg7qZCpyh/lBpNmPXNMQgh5C6NiIioThhUGojiUiMWbP4DAPDyfe3w/sMJ8HBTY29yFtYeSpO5OiIiorphUGkgfr+SB73BhGBvHR7uGoWoQE880bcFAOCrA5dkro6IiKhuGFQaiONpuQCADk18IUkSAGBEfCQA4LfkLOSXlMlWGxERUV0xqDQQN4KKn3VbbLAXmgd7wWAS+PnMdblKIyIiqjMGlQbiWHlQaVchqABAvzahAIBtpzKcXhMREVF9Mag0ACVlRpzJME9D7nBzUIkzB5Xtp6/BZOLsHyIici1c8M3FrNx7sdK21KwiGE0Cnlo1IvzcbV7r2iwQXlo1rhfoceJynrPKJCIisgu2qDQAl3OLAQBN/D2sA2kttBoVerUKBsDuHyIicj0MKg1AWrY5qET6e1T5+r1tLN0/DCpERORaGFQaAEuLSnVBpVerEADmAbd6g9FpdREREdUXg4qLMxhNuJprvrdPk2qCShN/DzTx94DRJJCaVezM8oiIiOqFQcXFXc3XwygEPNzUCPB0q3a/LrEBAICUzEJnlUZERFRvsgaVV155BZIk2TzatGkjZ0ku53r5nZLDfHWVBtJW1CU2EACQklnklLqIiIjsQfbpye3atcOPP/5ofa7RyF6SS8ktNi+N7++pveV+d5W3qFwsn8qsVlUfaoiIiJRC9lSg0WgQHh4u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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Applying Log transformations for Annual Income"
      ],
      "metadata": {
        "id": "GbPC0E-bNp6c"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.distplot(np.log(credit_card_raw['Annual_income']))\n",
        "plt.title(\"Distribution of Annual Income\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "id": "dlNGp3zuMgKY",
        "outputId": "13edbdae-5f16-4f45-dbd3-5e8ee72a8a4c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Distribution of Annual Income')"
            ]
          },
          "metadata": {},
          "execution_count": 179
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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H1mPhYaGipMmTbrlY96stkmTJomhoaHW65s2bRIBiO+8806l88aMGSMKgiBeuHDBegyAqFarKx07fvy4CED85JNPqjzX9ZYsWSICEL/66ivrMb1eL8bExIiurq6VXntoaKg4YsSImz7e9e69916xb9++1uuff/65qFKpxMzMzErnTZo0SQQgzps3r9Lx7t27iz179rReT0xMFAGI3t7eYm5urvX45s2bRQDi//73P+uxG7/31z/X9d9nURTF0tLSStf1er3YpUsX8e677650vKb3+UYAxGeffdZ6veL/UseOHUWdTmc9/tFHH4kAxJMnT4qiKIpGo1EMDw8XQ0NDxby8vEqPaTabrV9HRUWJfn5+Yk5OjvXY8ePHRYVCIU6cONF6rOLn58knn7QeMxqNYqtWrURBEMT333/fejwvL090cnKq9PrWrl0rKhQK8ffff69Uy/Lly0UA4r59+275vSD7xJYbouu4urredNaUh4cHAGDz5s31Hnyr0WgwZcqUWp8/ceJEuLm5Wa+PGTMGgYGB+Omnn+r1/LX1008/QalU4vnnn690/P/+7/8gimKVmUexsbFo27at9Xq3bt3g7u6OhISEWz5PQEAAHnnkEesxBwcHPP/88yguLsZvv/1Wr/pzcnKwffv2So/74IMPWrtnqvPUU09Vun7nnXdWW/+4cePg6elZ6TwAt3ytNakY4wUAeXl5KCgowJ133om4uLh6PV5NpkyZUmls1411Hzt2DImJiZg+fbr1/3qFiu7ZtLQ0xMfHY/LkyfDy8rLe3q1bNwwePLja/5dTp061fq1UKtGrVy+IoojHH3/cetzDwwPt27ev9D387rvv0LFjR3To0AHZ2dnWy9133w0A2L17d32/FSRzDDdE1ykuLq4UJG40btw49O3bF1OnToW/vz8efvhhbNiwoU5BJzg4uE6Dh9u1a1fpuiAIiIiIuOV4k4ZKTk5GUFBQle9Hx44drbdfr3Xr1lUew9PTE3l5ebd8nnbt2kGhqPxxVNPz1Nb69ethMBjQvXt3XLhwARcuXEBubi6io6Or7ZpydHSs0s1VU/03vtaKoHOr11qTLVu24I477oCjoyO8vLzg6+uLZcuWoaCgoF6PV5Nb1X3x4kUAQJcuXWp8jIr3o3379lVu69ixI7Kzs1FSUnLT59VqtXB0dISPj0+V49d/D8+fP49Tp07B19e30uW2224DYBmrRVQdzpYiuurSpUsoKChAREREjec4OTlh79692L17N7Zu3Ypt27Zh/fr1uPvuu/HLL79AqVTe8nmu/yvdVmpaaNBkMtWqJluo6XnEGwYfN5WKANO3b99qb09ISECbNm2s1+vyfarNaxUEodrXfv0AZQD4/fffMWrUKNx111347LPPEBgYCAcHB6xatQrr1q2rdU22qrsxVPe8tanFbDaja9euWLx4cbXnhoSE2KZAkh2GG6Kr1q5dCwAYOnToTc9TKBQYNGgQBg0ahMWLF+O9997D66+/jt27dyM2NtbmKxqfP3++0nVRFHHhwoVK6/F4enoiPz+/yn2Tk5Mr/QKvS22hoaHYuXMnioqKKrXenDlzxnq7LYSGhuLEiRMwm82VWm8a8jyJiYnYv38/pk2bhv79+1e6zWw2Y8KECVi3bh3eeOONhhV/E56entV2U93YEvXf//4Xjo6O2L59OzQajfX4qlWrGq22mlR0K/7111+IjY2t9pyK9+Ps2bNVbjtz5gx8fHxsNtW+bdu2OH78OAYNGtRkK4WTPLBbigiWaa1vv/02wsPDq8weuV5ubm6VYxWL4VVMTa34YK8ubNTHmjVrKo0D+v7775GWllZpMbq2bdvizz//hF6vtx7bsmVLlSnjdalt+PDhMJlM+PTTTysd//DDDyEIgs0Wwxs+fDjS09Oxfv166zGj0YhPPvkErq6uVcJJbVS02rz88ssYM2ZMpcvYsWPRv3//GmdN2Urbtm1x5swZZGVlWY8dP37cOrOuglKphCAIVaacb9q0qVHrq06PHj0QHh6OJUuWVPk/UtGiEhgYiKioKHz55ZeVzvnrr7/wyy+/YPjw4TarZ+zYsbh8+TJWrFhR5baysrIq3V9EFdhyQ3bn559/xpkzZ2A0GpGRkYFff/0VO3bsQGhoKH788Uc4OjrWeN958+Zh7969GDFiBEJDQ5GZmYnPPvsMrVq1Qr9+/QBYfql5eHhg+fLlcHNzg4uLC6KjoxEeHl6ver28vNCvXz9MmTIFGRkZWLJkCSIiIvDEE09Yz5k6dSq+//573HPPPRg7diwuXryIr776qtIA37rWNnLkSAwcOBCvv/46kpKSEBkZiV9++QWbN2/G9OnTqzx2fT355JP497//jcmTJ+Po0aMICwvD999/j3379mHJkiU3HQNVk6+//hpRUVE1dluMGjUKzz33HOLi4tCjR4+GvoRqPfbYY1i8eDGGDh2Kxx9/HJmZmVi+fDk6d+6MwsJC63kjRozA4sWLcc8992D8+PHIzMzE0qVLERERgRMnTjRKbTVRKBRYtmwZRo4ciaioKEyZMgWBgYE4c+YMTp06he3btwMAPvjgAwwbNgwxMTF4/PHHUVZWhk8++QRarRZvvvmmzeqZMGECNmzYgKeeegq7d+9G3759YTKZcObMGWzYsAHbt2+vtCAnkZVk87SImljFVPCKi1qtFgMCAsTBgweLH330UaUpxxVunAq+a9cu8b777hODgoJEtVotBgUFiY888oh47ty5SvfbvHmz2KlTJ1GlUlWaet2/f3+xc+fO1dZX01Twb775Rpw1a5bo5+cnOjk5iSNGjBCTk5Or3H/RokVicHCwqNFoxL59+4pHjhypdjpyTbVVN0W5qKhIfPHFF8WgoCDRwcFBbNeunfjBBx9UmhYsilWnHleo7dTljIwMccqUKaKPj4+oVqvFrl27VjtdvTZTwY8ePSoCEGfPnl3jOUlJSSIA8cUXXxRF0fLaXVxcqpx34/tfMRX8gw8+qHIuAHHu3LmVjn311VdimzZtRLVaLUZFRYnbt2+v9vu8cuVKsV27dqJGoxE7dOggrlq1qspzV7z+hkwF/+677yqdV/F6bvxe//HHH+LgwYNFNzc30cXFRezWrVuVKf07d+4U+/btKzo5OYnu7u7iyJEjxb///rvSORWv4calE2r6flf386HX68UFCxaInTt3FjUajejp6Sn27NlTfOutt8SCgoJbfi/IPgmiKNFoPyIiIqJGwDE3REREJCsMN0RERCQrDDdEREQkKww3REREJCsMN0RERCQrDDdEREQkK3a3iJ/ZbMaVK1fg5ubG5byJiIhaCFEUUVRUhKCgoCob7d7I7sLNlStXuNkaERFRC5WamopWrVrd9By7CzcVS7mnpqbC3d1d4mqIiIioNgoLCxESElKrLVnsLtxUdEW5u7sz3BAREbUwtRlSwgHFREREJCsMN0RERCQrDDdEREQkKww3REREJCsMN0RERCQrDDdEREQkKww3REREJCsMN0RERCQrDDdEREQkKww3REREJCsMN0RERCQrDDdEREQkKww3REREJCsMN0RERCQrkoabvXv3YuTIkQgKCoIgCNi0adNNz9+4cSMGDx4M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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.distplot(credit_card_raw['age'])\n",
        "plt.title(\"Distribution of age\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "id": "9WMbwYSrQ4l1",
        "outputId": "c5bfc201-7d21-47f6-ebe4-178f3f164b4f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Distribution of age')"
            ]
          },
          "metadata": {},
          "execution_count": 180
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkgAAAHHCAYAAABEEKc/AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABpBElEQVR4nO3deVxVdf7H8de9wAXZEWQVBFc0F1wJ0ywltWyxZTJrshxbp91ZyqasaRnbLGtqcpylpimzsV9jZmYZWo1JLrivuYLKJiK7rPf8/kBugaiIwOHC+/l43Edy7vee+7knhLff7VgMwzAQEREREQer2QWIiIiItDYKSCIiIiJ1KCCJiIiI1KGAJCIiIlKHApKIiIhIHQpIIiIiInUoIImIiIjUoYAkIiIiUocCkoiIiEgdCkgiclpPP/00FoulRd7rkksu4ZJLLnF8/c0332CxWPj4449b5P1vv/12oqOjW+S9GquoqIg77riD0NBQLBYLDz/8sNklibRZCkgi7cS7776LxWJxPDw8PAgPD2fcuHG88cYbFBYWNsn7pKen8/TTT7Np06YmOV9Tas21NcSf/vQn3n33Xe69917+/e9/c+utt5pdkkibZdG92ETah3fffZepU6fyzDPPEBMTQ0VFBZmZmXzzzTcsX76cqKgoFi9eTP/+/R2vqayspLKyEg8Pjwa/z/r16xk6dCjvvPMOt99+e4NfV15eDoDNZgOqe5AuvfRSFi5cyA033NDg8zS2toqKCux2O+7u7k3yXs3hwgsvxNXVlVWrVpldikib52p2ASLSsi6//HKGDBni+HrGjBmsWLGCK6+8kquvvpqdO3fSoUMHAFxdXXF1bd4fEyUlJXh6ejqCkVnc3NxMff+GyM7Opk+fPmaXIdIuaIhNRBg9ejRPPvkkqampvP/++47j9c1BWr58OSNGjMDf3x9vb2969erF448/DlT3+gwdOhSAqVOnOobz3n33XaB6nlHfvn1JSUnh4osvxtPT0/HaunOQalRVVfH4448TGhqKl5cXV199NYcOHarVJjo6ut7eqp+f82y11TcHqbi4mN/85jdERkbi7u5Or169eOWVV6jb8W6xWLj//vtZtGgRffv2xd3dnQsuuIBly5bVf8HryM7OZtq0aYSEhODh4cGAAQP417/+5Xi+Zj7WgQMH+Pzzzx21Hzx48LTnfOeddxg9ejTBwcG4u7vTp08f3n777VPa2e12nn76acLDw/H09OTSSy9lx44d9V7TvLw8Hn74Ycf16N69Oy+++CJ2u71Bn1PEmagHSUQAuPXWW3n88cf56quvuPPOO+tts337dq688kr69+/PM888g7u7O3v37uX7778HoHfv3jzzzDPMnDmTu+66i5EjRwIwfPhwxzmOHTvG5Zdfzk033cQvf/lLQkJCzljX888/j8Vi4dFHHyU7O5s5c+aQmJjIpk2bHD1dDdGQ2n7OMAyuvvpqVq5cybRp04iLi+PLL7/kd7/7HUeOHOG1116r1X7VqlV88skn/PrXv8bHx4c33niD66+/nrS0NAIDA09b14kTJ7jkkkvYu3cv999/PzExMSxcuJDbb7+dvLw8HnroIXr37s2///1vHnnkETp37sxvfvMbADp16nTa87799ttccMEFXH311bi6uvLZZ5/x61//Grvdzn333edoN2PGDF566SWuuuoqxo0bx+bNmxk3bhylpaW1zldSUsKoUaM4cuQId999N1FRUaxevZoZM2aQkZHBnDlzznj9RZyOISLtwjvvvGMAxrp1607bxs/Pzxg4cKDj66eeesr4+Y+J1157zQCMo0ePnvYc69atMwDjnXfeOeW5UaNGGYAxd+7cep8bNWqU4+uVK1cagBEREWEUFBQ4jv/nP/8xAOP11193HOvSpYtx2223nfWcZ6rttttuM7p06eL4etGiRQZgPPfcc7Xa3XDDDYbFYjH27t3rOAYYNput1rHNmzcbgPHnP//5lPf6uTlz5hiA8f777zuOlZeXGwkJCYa3t3etz96lSxdjwoQJZzxfjZKSklOOjRs3zujatavj68zMTMPV1dWYOHFirXZPP/20AdS6ps8++6zh5eVl/Pjjj7XaPvbYY4aLi4uRlpbWoLpEnIWG2ETEwdvb+4yr2fz9/QH49NNPGz2s4u7uztSpUxvcfsqUKfj4+Di+vuGGGwgLC2Pp0qWNev+GWrp0KS4uLjz44IO1jv/mN7/BMAy++OKLWscTExPp1q2b4+v+/fvj6+vL/v37z/o+oaGhTJ482XHMzc2NBx98kKKiIr799ttG1f/z3rX8/HxycnIYNWoU+/fvJz8/H4CkpCQqKyv59a9/Xeu1DzzwwCnnW7hwISNHjiQgIICcnBzHIzExkaqqKr777rtG1SnSWikgiYhDUVFRrTBS16RJk7jooou44447CAkJ4aabbuI///nPOYWliIiIc5qQ3aNHj1pfWywWunfvfsb5N00hNTWV8PDwU65H7969Hc//XFRU1CnnCAgI4Pjx42d9nx49emC11v5xfLr3aajvv/+exMREvLy88Pf3p1OnTo75XjUBqebc3bt3r/Xajh07EhAQUOvYnj17WLZsGZ06dar1SExMBKrnUYm0JZqDJCIAHD58mPz8/FN+Wf5chw4d+O6771i5ciWff/45y5Yt46OPPmL06NF89dVXuLi4nPV9zmXeUEOdbjPLqqqqBtXUFE73PoYJO6ns27ePMWPGEBsby6uvvkpkZCQ2m42lS5fy2muvNar3z263c9lll/H73/++3ud79ux5vmWLtCoKSCICwL///W8Axo0bd8Z2VquVMWPGMGbMGF599VX+9Kc/8Yc//IGVK1eSmJjY5Dtv79mzp9bXhmGwd+/eWvs1BQQEkJeXd8prU1NT6dq1q+Prc6mtS5cufP311xQWFtbqRdq1a5fj+abQpUsXtmzZgt1ur9WLdD7v89lnn1FWVsbixYtr9WytXLnylPcG2Lt3LzExMY7jx44dO6Xnq1u3bhQVFTl6jETaOg2xiQgrVqzg2WefJSYmhltuueW07XJzc085FhcXB0BZWRkAXl5eAPUGlsZ47733as2L+vjjj8nIyODyyy93HOvWrRs//PCDY7NJgCVLlpyyHcC51HbFFVdQVVXFm2++Wev4a6+9hsViqfX+5+OKK64gMzOTjz76yHGssrKSP//5z3h7ezNq1KhzPmdNb9bPe6/y8/N55513arUbM2YMrq6upyz/r/uZAW688UaSk5P58ssvT3kuLy+PysrKc65TpDVTD5JIO/PFF1+wa9cuKisrycrKYsWKFSxfvpwuXbqwePHiM+6a/cwzz/Ddd98xYcIEunTpQnZ2Nn/5y1/o3LkzI0aMAKrDir+/P3PnzsXHxwcvLy/i4+Nr9VCci44dOzJixAimTp1KVlYWc+bMoXv37rW2Irjjjjv4+OOPGT9+PDfeeCP79u3j/fffrzVp+lxru+qqq7j00kv5wx/+wMGDBxkwYABfffUVn376KQ8//PAp526su+66i7/+9a/cfvvtpKSkEB0dzccff8z333/PnDlzzjgn7HTGjh2LzWbjqquu4u6776aoqIi//e1vBAcHk5GR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6NGjzJw5k8zMTOLi4li2bJljknVaWhpW60/LA4cPH878+fN54oknePzxx+nRoweLFi1y7IEE8Pvf/57i4mLuuusu8vLyGDFiBMuWLXPsgQTwwQcfcP/99zNmzBisVivXX389b7zxRst9cBEREWm1TN8HSdqesrIyZs2axYwZM04ZlhRdnzPRtTk9XZsz0/U5PV2bxlFAEhEREamj9WxtKSIiItJKKCCJiIiI1KGAJCIiIlKHApKIiIhIHQpI0iizZs1i6NCh+Pj4EBwczMSJE9m9e3etNqWlpdx3330EBgbi7e3N9ddff8ou6G3V22+/Tf/+/R0bsyUkJPDFF184nm/P16auF154wbF/WY32fH2efvppLBZLrUdsbKzj+fZ8bQCOHDnCL3/5SwIDA+nQoQP9+vVj/fr1jucNw2DmzJmEhYXRoUMHEhMT2bNnj4kVt5zo6OhTvncsFgv33XcfoO+dc6WAJI3y7bffct999/HDDz+wfPlyKioqGDt2LMXFxY42jzzyCJ999hkLFy7k22+/JT09neuuu87EqltO586deeGFF0hJSWH9+vWMHj2aa665hu3btwPt+9r83Lp16/jrX/9K//79ax1v79fnggsuICMjw/FYtWqV47n2fG2OHz/ORRddhJubG1988QU7duxg9uzZBAQEONq89NJLvPHGG8ydO5c1a9bg5eXFuHHjKC0tNbHylrFu3bpa3zfLly8H4Be/+AXQvr93GsUQaQLZ2dkGYHz77beGYRhGXl6e4ebmZixcuNDRZufOnQZgJCcnm1WmqQICAoy///3vujYnFRYWGj169DCWL19ujBo1ynjooYcMw9D3zlNPPWUMGDCg3ufa+7V59NFHjREjRpz2ebvdboSGhhovv/yy41heXp7h7u5ufPjhhy1RYqvy0EMPGd26dTPsdnu7/95pDPUgSZPIz88HoGPHjgCkpKRQUVFBYmKio01sbCxRUVEkJyebUqNZqqqqWLBgAcXFxSQkJOjanHTfffcxYcKEWtcB9L0DsGfPHsLDw+natSu33HILaWlpgK7N4sWLGTJkCL/4xS8IDg5m4MCB/O1vf3M8f+DAATIzM2tdHz8/P+Lj49vF9fm58vJy3n//fX71q19hsVja/fdOYyggyXmz2+08/PDDXHTRRY5bvmRmZmKz2U65oW9ISAiZmZkmVNnytm7dire3N+7u7txzzz3897//pU+fPro2wIIFC9iwYYPjHos/196vT3x8PO+++y7Lli3j7bff5sCBA4wcOZLCwsJ2f23279/P22+/TY8ePfjyyy+59957efDBB/nXv/4F4LgGNbeqqtFers/PLVq0iLy8PG6//XZAf68aw/R7sYnzu++++9i2bVuteRICvXr1YtOmTeTn5/Pxxx9z22238e2335pdlukOHTrEQw89xPLly2vdH1GqXX755Y4/9+/fn/j4eLp06cJ//vMfOnToYGJl5rPb7QwZMoQ//elPAAwcOJBt27Yxd+5cbrvtNpOra13+8Y9/cPnll+sG7OdBPUhyXu6//36WLFnCypUr6dy5s+N4aGgo5eXl5OXl1WqflZVFaGhoC1dpDpvNRvfu3Rk8eDCzZs1iwIABvP766+3+2qSkpJCdnc2gQYNwdXXF1dWVb7/9ljfeeANXV1dCQkLa9fWpy9/fn549e7J37952/70TFhZGnz59ah3r3bu3Ywiy5hrUXZnVXq5PjdTUVL7++mvuuOMOx7H2/r3TGApI0iiGYXD//ffz3//+lxUrVhATE1Pr+cGDB+Pm5kZSUpLj2O7du0lLSyMhIaGly20V7HY7ZWVl7f7ajBkzhq1bt7Jp0ybHY8iQIdxyyy2OP7fn61NXUVER+/btIywsrN1/71x00UWnbCfy448/0qVLFwBiYmIIDQ2tdX0KCgpYs2ZNu7g+Nd555x2Cg4OZMGGC41h7/95pFLNniYtzuvfeew0/Pz/jm2++MTIyMhyPkpISR5t77rnHiIqKMlasWGGsX7/eSEhIMBISEkysuuU89thjxrfffmscOHDA2LJli/HYY48ZFovF+OqrrwzDaN/Xpj4/X8VmGO37+vzmN78xvvnmG+PAgQPG999/byQmJhpBQUFGdna2YRjt+9qsXbvWcHV1NZ5//nljz549xgcffGB4enoa77//vqPNCy+8YPj7+xuffvqpsWXLFuOaa64xYmJijBMnTphYecupqqoyoqKijEcfffSU59rz905jKCBJowD1Pt555x1HmxMnThi//vWvjYCAAMPT09O49tprjYyMDPOKbkG/+tWvjC5duhg2m83o1KmTMWbMGEc4Moz2fW3qUzcgtefrM2nSJCMsLMyw2WxGRESEMWnSJGPv3r2O59vztTEMw/jss8+Mvn37Gu7u7kZsbKwxb968Ws/b7XbjySefNEJCQgx3d3djzJgxxu7du02qtuV9+eWXBlDvZ27v3zvnymIYhmFiB5aIiIhIq6M5SCIiIiJ1KCCJiIiI1KGAJCIiIlKHApKIiIhIHQpIIiIiInUoIImIiIjUoYAkIiIiUocCkoiIiEgdCkgiIiIidSggiYiIiNShgCQi7cayZcsYMWIE/v7+BAYGcuWVV7Jv3z7H86tXryYuLg4PDw+GDBnCokWLsFgsbNq0ydFm27ZtXH755Xh7exMSEsKtt95KTk6OCZ9GRJqTApKItBvFxcVMnz6d9evXk5SUhNVq5dprr8Vut1NQUMBVV11Fv3792LBhA88++yyPPvpordfn5eUxevRoBg4cyPr161m2bBlZWVnceOONJn0iEWkuulmtiLRbOTk5dOrUia1bt7Jq1SqeeOIJDh8+jIeHBwB///vfufPOO9m4cSNxcXE899xz/O9//+PLL790nOPw4cNERkaye/duevbsadZHEZEmph4kEWk39uzZw+TJk+natSu+vr5ER0cDkJaWxu7du+nfv78jHAEMGzas1us3b97MypUr8fb2djxiY2MBag3ViYjzczW7ABGRlnLVVVfRpUsX/va3vxEeHo7dbqdv376Ul5c36PVFRUVcddVVvPjii6c8FxYW1tTlioiJFJBEpF04duwYu3fv5m9/+xsjR44EYNWqVY7ne/Xqxfvvv09ZWRnu7u4ArFu3rtY5Bg0axP/93/8RHR2Nq6t+fIq0ZRpiE5F2ISAggMDAQObNm8fevXtZsWIF06dPdzx/8803Y7fbueuuu9i5cydffvklr7zyCgAWiwWA++67j9zcXCZPnsy6devYt28fX375JVOnTqWqqsqUzyUizUMBSUTaBavVyoIFC0hJSaFv37488sgjvPzyy47nfX19+eyzz9i0aRNxcXH84Q9/YObMmQCOeUnh4eF8//33VFVVMXbsWPr168fDDz+Mv78/Vqt+nIq0JVrFJiJyGh988AFTp04lPz+fDh06mF2OiLQgDaKLiJz03nvv0bVrVyIiIti8eTOPPvooN954o8KRSDukgCQiclJmZiYzZ84kMzOTsLAwfvGLX/D888+bXZaImEBDbCIiIiJ1aFahiIiISB0KSCIiIiJ1KCCJiIiI1KGAJCIiIlKHApKIiIhIHQpIIiIiInUoIImIiIjUoYAkIiIiUocCkoiIiEgd/w/xGNnj6U+kugAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw=credit_card_raw.drop(['Mobile_phone'],axis=1)"
      ],
      "metadata": {
        "id": "NwG2HG0BYRCn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "corr=credit_card_raw.corr()\n",
        "plt.figure(figsize=(10,6))\n",
        "sns.heatmap(corr,annot=True)"
      ],
      "metadata": {
        "id": "pde7L8ilbZZz",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 647
        },
        "outputId": "7feecdc3-7407-4949-9faa-6784d2270347"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: >"
            ]
          },
          "metadata": {},
          "execution_count": 182
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Feature Engineering"
      ],
      "metadata": {
        "id": "P4jIeLy_a4FG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Ordinal Encoding"
      ],
      "metadata": {
        "id": "rNgjVPGQROZl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import OrdinalEncoder\n",
        "\n",
        "oe=OrdinalEncoder()\n",
        "credit_card_raw[['Type_Income']]=oe.fit_transform(credit_card_raw[['Type_Income']])\n",
        "credit_card_raw[['EDUCATION']]=oe.fit_transform(credit_card_raw[['EDUCATION']])\n",
        "credit_card_raw[['Housing_type']]=oe.fit_transform(credit_card_raw[['Housing_type']])\n",
        "credit_card_raw[['Type_Occupation']]=oe.fit_transform(credit_card_raw[['Type_Occupation']])\n"
      ],
      "metadata": {
        "id": "RqIIfna0YmfT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# One Hot Coding"
      ],
      "metadata": {
        "id": "EOoQClkdRTzp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw=pd.get_dummies(credit_card_raw,columns=['Car_Owner',\t'Propert_Owner','Marital_status','GENDER'],drop_first=True)"
      ],
      "metadata": {
        "id": "O2MQ7TIDZyoP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "credit_card_raw.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 273
        },
        "id": "1WHBTC_pLunh",
        "outputId": "196728cc-b1dd-4d28-a25e-4c04c8756f95"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "    Ind_ID  CHILDREN  Annual_income  Type_Income  EDUCATION  Housing_type  \\\n",
              "0  5008827         0   180000.00000          1.0        1.0           1.0   \n",
              "1  5009744         0   315000.00000          0.0        1.0           1.0   \n",
              "2  5009746         0   315000.00000          0.0        1.0           1.0   \n",
              "3  5009749         0   191399.32623          0.0        1.0           1.0   \n",
              "4  5009752         0   315000.00000          0.0        1.0           1.0   \n",
              "\n",
              "   Work_Phone  Phone  EMAIL_ID  Type_Occupation  ...  label        age  \\\n",
              "0           0      0         0              8.0  ...      1  51.430137   \n",
              "1           1      1         0              8.0  ...      1  37.142466   \n",
              "2           1      1         0              8.0  ...      1  43.946143   \n",
              "3           1      1         0              8.0  ...      1  37.142466   \n",
              "4           1      1         0              8.0  ...      1  37.142466   \n",
              "\n",
              "    experience  Car_Owner_Y  Propert_Owner_Y  Marital_status_Married  \\\n",
              "0  1000.665753            1                1                       1   \n",
              "1     1.605479            1                0                       1   \n",
              "2     1.605479            1                0                       1   \n",
              "3     1.605479            1                0                       1   \n",
              "4     1.605479            1                0                       1   \n",
              "\n",
              "   Marital_status_Separated  Marital_status_Single / not married  \\\n",
              "0                         0                                    0   \n",
              "1                         0                                    0   \n",
              "2                         0                                    0   \n",
              "3                         0                                    0   \n",
              "4                         0                                    0   \n",
              "\n",
              "   Marital_status_Widow  GENDER_M  \n",
              "0                     0         1  \n",
              "1                     0         0  \n",
              "2                     0         0  \n",
              "3                     0         0  \n",
              "4                     0         0  \n",
              "\n",
              "[5 rows x 21 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e6faf849-06e8-49db-96c1-4ffd1f87023d\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ind_ID</th>\n",
              "      <th>CHILDREN</th>\n",
              "      <th>Annual_income</th>\n",
              "      <th>Type_Income</th>\n",
              "      <th>EDUCATION</th>\n",
              "      <th>Housing_type</th>\n",
              "      <th>Work_Phone</th>\n",
              "      <th>Phone</th>\n",
              "      <th>EMAIL_ID</th>\n",
              "      <th>Type_Occupation</th>\n",
              "      <th>...</th>\n",
              "      <th>label</th>\n",
              "      <th>age</th>\n",
              "      <th>experience</th>\n",
              "      <th>Car_Owner_Y</th>\n",
              "      <th>Propert_Owner_Y</th>\n",
              "      <th>Marital_status_Married</th>\n",
              "      <th>Marital_status_Separated</th>\n",
              "      <th>Marital_status_Single / not married</th>\n",
              "      <th>Marital_status_Widow</th>\n",
              "      <th>GENDER_M</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>5008827</td>\n",
              "      <td>0</td>\n",
              "      <td>180000.00000</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>...</td>\n",
              "      <td>1</td>\n",
              "      <td>51.430137</td>\n",
              "      <td>1000.665753</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>5009744</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.00000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>...</td>\n",
              "      <td>1</td>\n",
              "      <td>37.142466</td>\n",
              "      <td>1.605479</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>5009746</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.00000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>...</td>\n",
              "      <td>1</td>\n",
              "      <td>43.946143</td>\n",
              "      <td>1.605479</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>5009749</td>\n",
              "      <td>0</td>\n",
              "      <td>191399.32623</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>...</td>\n",
              "      <td>1</td>\n",
              "      <td>37.142466</td>\n",
              "      <td>1.605479</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5009752</td>\n",
              "      <td>0</td>\n",
              "      <td>315000.00000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>8.0</td>\n",
              "      <td>...</td>\n",
              "      <td>1</td>\n",
              "      <td>37.142466</td>\n",
              "      <td>1.605479</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 21 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e6faf849-06e8-49db-96c1-4ffd1f87023d')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e6faf849-06e8-49db-96c1-4ffd1f87023d button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e6faf849-06e8-49db-96c1-4ffd1f87023d');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-854c9f41-0b8f-4196-be59-6f6a5fcdc9fa\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-854c9f41-0b8f-4196-be59-6f6a5fcdc9fa')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-854c9f41-0b8f-4196-be59-6f6a5fcdc9fa button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 65
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Splitting the data"
      ],
      "metadata": {
        "id": "aLJ27xXaDBaX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X=credit_card_raw.drop(['label'],axis=1)\n",
        "y=credit_card_raw['label']"
      ],
      "metadata": {
        "id": "WaZFI7JCdl9R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Feature Selection"
      ],
      "metadata": {
        "id": "FtC2E0AEd8hk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.feature_selection import RFE\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "# Create a Random Forest classifier\n",
        "rf_classifier = RandomForestClassifier(n_estimators=10, random_state=10)\n",
        "\n",
        "# Create an RFE object using the Random Forest classifier as the base estimator\n",
        "rfe_method = RFE(rf_classifier, n_features_to_select=8, step=2)\n",
        "\n",
        "# Fit the RFE object to the data\n",
        "rfe_method.fit(X, y)\n",
        "\n",
        "# Print the number of features selected\n",
        "print(\"Num Features: %d\" % rfe_method.n_features_)\n",
        "\n",
        "# Print the selected features\n",
        "print(\"Selected Features: %s\" % rfe_method.support_)\n",
        "\n",
        "# Print the feature ranking\n",
        "print(\"Feature Ranking: %s\" % rfe_method.ranking_)"
      ],
      "metadata": {
        "id": "4M_n4-g7Af3f",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "84e1930b-7c5f-4efd-905a-0e9d4c563634"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Num Features: 8\n",
            "Selected Features: [ True False  True False  True  True False False False  True  True  True\n",
            "  True False False False False False False False]\n",
            "Feature Ranking: [1 5 1 3 1 1 5 3 6 1 1 1 1 2 2 6 7 4 7 4]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X.columns"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OvkZ9myrFmz3",
        "outputId": "dac08f56-03fb-4c41-9a0b-66c9898495c8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['Ind_ID', 'CHILDREN', 'Annual_income', 'Type_Income', 'EDUCATION',\n",
              "       'Housing_type', 'Work_Phone', 'Phone', 'EMAIL_ID', 'Type_Occupation',\n",
              "       'Family_Members', 'age', 'experience', 'Car_Owner_Y', 'Propert_Owner_Y',\n",
              "       'Marital_status_Married', 'Marital_status_Separated',\n",
              "       'Marital_status_Single / not married', 'Marital_status_Widow',\n",
              "       'GENDER_M'],\n",
              "      dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 68
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X1=X[['Ind_ID','Annual_income','Type_Income', 'EDUCATION','Family_Members','Type_Occupation', 'age', 'experience']]\n",
        "y=credit_card_raw[\"label\"]"
      ],
      "metadata": {
        "id": "6uGFS27neH8o"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Outlier Detection and Removal"
      ],
      "metadata": {
        "id": "Az5FtQUWRcw1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for i in X1:\n",
        "  sns.boxplot(X1[i])\n",
        "  plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "FU41YgT_fXK9",
        "outputId": "bfa34636-1256-45d3-cc9d-204e0d597aa7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiMAAAGsCAYAAAAPJKchAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAmtUlEQVR4nO3df3DU1b3/8dduIrtwIevFyAJpYqwt/rhAkgaSButcsdE0tbkDnbb5EjRMrPaqwM01Y1tSNblqa/xFCreGRin+YEoAdRqmlx9RmjtexpIRCOZbuNdKLegGYRfQy24STCK7+/2DLytbEsgC2ZPdfT5mPjPkfM7Zfe8wYV+cz/mcjyUYDAYFAABgiNV0AQAAILERRgAAgFGEEQAAYBRhBAAAGEUYAQAARhFGAACAUYQRAABgFGEEAAAYRRgBAABGEUYAAIBRMRVGtm3bppKSEk2ePFkWi0UbNmyI+DWCwaCeffZZTZkyRTabTWlpafrFL35x6YsFAABDkmy6gEj09PQoKytLd911l7773e9e0GtUVlbqzTff1LPPPqtp06bp008/1aeffnqJKwUAAENlidUH5VksFjU3N2vOnDmhtr6+Pj300ENau3atjh8/rqlTp+qpp57SzTffLEl67733NH36dO3du1fXXnutmcIBAECYmLpMcz6LFi1SW1ub1q1bpz/96U/6/ve/r29961v6y1/+Ikn6j//4D335y1/Wxo0bdfXVVyszM1N33303MyMAABgUN2HE5XLppZde0muvvaabbrpJ11xzjR588EF94xvf0EsvvSRJ2r9/vz766CO99tprWr16tV5++WW1t7fre9/7nuHqAQBIXDG1ZuRc9uzZI7/frylTpoS19/X16YorrpAkBQIB9fX1afXq1aF+q1atUm5urt5//30u3QAAYEDchJHu7m4lJSWpvb1dSUlJYefGjh0rSZo0aZKSk5PDAsv1118v6dTMCmEEAIDoi5swkpOTI7/fryNHjuimm24asM+NN96okydP6q9//auuueYaSdK+ffskSVdddVXUagUAAF+Iqbtpuru79cEHH0g6FT7q6+s1e/ZsjR8/XhkZGbrjjjv0xz/+UUuXLlVOTo6OHj2q1tZWTZ8+XbfffrsCgYBmzpypsWPHatmyZQoEAlq4cKFSUlL05ptvGv50AAAkppgKI2+99ZZmz559VvuC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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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5s6d2/AJT548mVdffTWzZ89ueC0AMPU0dGVk3bp12bx5c5555pnMnDkz/f39SZJqtZoZM2YkSbq6ujJnzpz09vYmSb73ve9l8eLFue666/KXv/wljzzySN58883ce++9Z/mpAACTUUMx8vjjjydJli1bNmb/T3/60/zDP/xDkuTw4cOZNu2vF1z+/Oc/57777kt/f3+uuOKKLFy4MHv27MkNN9zw6SYHAKaEhmKkXq9/4jG7d+8e8/jRRx/No48+2tBQAMDFw3fTAABFiREAoCgxAgAUJUYAgKLECABQlBgBAIoSIwBAUWIEAChKjAAARYkRAKAoMQIAFCVGAICixAgAUJQYAQCKEiMAQFFiBAAoSowAAEWJEQCgKDECABQlRgCAosQIAFCUGAEAihIjAEBRYgQAKEqMAABFiREAoCgxAgAUJUYAgKLECABQlBgBAIoSIwBAUWIEAChKjAAARYkRAKAoMQIAFCVGAICixAgAUJQYAQCKEiMAQFFiBAAoSowAAEWJEQCgKDECABQlRgCAosQIAFBUQzHS29ubW2+9NTNnzsysWbOyatWqvP7665+4buvWrbn++uvT3Nycm266Kc8+++yEBwYAppaGYuTFF1/MunXr8vvf/z67du3KBx98kK9//esZGhoad82ePXty11135Z577skrr7ySVatWZdWqVTl06NCnHh4AmPya6vV6faKL33nnncyaNSsvvvhivvzlL5/ymDVr1mRoaCg7duwY3bd48eJ0dHRk48aNZ3SeWq2WarWawcHBtLS0THTcYt57772sXLkySXLsi3+fTL+08EQAXLBOfpCZB36eJNm5c2dmzJhReKCJO9PX7091z8jg4GCS5Morrxz3mL1792b58uVj9q1YsSJ79+4dd83w8HBqtdqYDQCYmiYcIyMjI3nwwQfzpS99KTfeeOO4x/X396e1tXXMvtbW1vT394+7pre3N9VqdXRrb2+f6JgAwAVuwjGybt26HDp0KFu2bDmb8yRJenp6Mjg4OLodOXLkrJ8DALgwXDKRRffff3927NiRl156Kddcc81pj21ra8vAwMCYfQMDA2lraxt3TaVSSaVSmchoAMAk09CVkXq9nvvvvz/btm3LCy+8kLlz537ims7OzvT19Y3Zt2vXrnR2djY2KQAwJTV0ZWTdunXZvHlznnnmmcycOXP0vo9qtTp6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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiMAAAGdCAYAAADAAnMpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAeFElEQVR4nO3df0xV9/3H8dfFH/fSlXuna+Ui3rYmNlqrguIPLk2q3WgJMY30D0PMUpxRky7Y6FiylKbRrOtymxinTWpF0zq3NnxxtgUTJzpGg6aDrqKSqMtM3BqhLRdt0t6LRK+Oe79/9Nvb3a+CHEDfgM9HcpLew/lw3jeLu8+ce7jXlUgkEgIAADCSZj0AAAC4txEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADA1HjrAQYiHo/ryy+/VEZGhlwul/U4AABgABKJhLq7uzV16lSlpfV9/WNUxMiXX36pQCBgPQYAABiEjo4OTZs2rc+fj4oYycjIkPTtk/F6vcbTAACAgYhGowoEAsnX8b6Mihj57q0Zr9dLjAAAMMrc7hYLbmAFAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaGFCOvv/66XC6XNm3a1O9xBw4c0KxZs+TxeDR37lwdPnx4KKcFAABjyKBj5MSJE9q9e7fmzZvX73HNzc1atWqV1q5dq9OnT6ukpEQlJSU6e/bsYE8NAADGEFcikUg4XXTlyhUtWLBAb731ll577TXl5uZqx44dtzy2tLRUPT09OnToUHJffn6+cnNzVVVVNaDzRaNR+Xw+RSKRUfndNPF4XJFIxHoM/J9EIqFYLGY9BjBiud3u236XCO4On8+ntLTRe0fFQF+/B/VFeeXl5Vq+fLkKCwv12muv9XtsS0uLKioqUvYVFRWprq6uzzWxWCzlxSIajQ5mzBEjEonoueeesx4DADDK1NbWatKkSdZj3HGOY6SmpkanTp3SiRMnBnR8OBxWZmZmyr7MzEyFw+E+14RCIf361792OhoAABiFHMVIR0eHNm7cqIaGBnk8njs1kyorK1OupkSjUQUCgTt2vjvN7XYn//vK3JVKpI0znAZKSIr/x3oKYORKGy/xLo0ZV7xX9585ICn19WMscxQjJ0+e1KVLl7RgwYLkvt7eXh0/flxvvvmmYrGYxo1LfaH1+/3q6upK2dfV1SW/39/nedxu95j6H+C/33tNTPBI4yYYTgMAGMkSvTeS/32v3Lvj6K6Yn/zkJzpz5oza2tqS28KFC/XTn/5UbW1tN4WIJAWDQTU2Nqbsa2hoUDAYHNrkAABgTHB0ZSQjI0Nz5sxJ2feDH/xAP/rRj5L7y8rKlJ2drVAoJEnauHGjli5dqm3btmn58uWqqalRa2ur9uzZM0xPAQAAjGbD/vdC7e3t6uzsTD4uKChQdXW19uzZo5ycHL3//vuqq6u7KWoAAMC9aVB/2vvfmpqa+n0sSStXrtTKlSuHeioAADAGjd5PUgEAAGMCMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMCUoxjZtWuX5s2bJ6/XK6/Xq2AwqPr6+j6P37dvn1wuV8rm8XiGPDQAABg7xjs5eNq0aXr99df16KOPKpFI6A9/+INWrFih06dP6/HHH7/lGq/Xq/Pnzycfu1yuoU0MAADGFEcx8uyzz6Y8/u1vf6tdu3bpk08+6TNGXC6X/H7/4CcEAABj2qDvGent7VVNTY16enoUDAb7PO7KlSt6+OGHFQgEtGLFCp07d+62vzsWiykajaZsAABgbHIcI2fOnNH9998vt9utF154QbW1tZo9e/Ytj505c6b27t2rgwcP6r333lM8HldBQYE+//zzfs8RCoXk8/mSWyAQcDomAAAYJVyJRCLhZMH169fV3t6uSCSi999/X2+//baOHTvWZ5D8txs3buixxx7TqlWr9Jvf/KbP42KxmGKxWPJxNBpVIBBQJBKR1+t1Mu6IcPXqVRUXF0uSuhc8L42bYDwRAGDE6r2hjFPvSpLq6+uVnp5uPNDgRaNR+Xy+275+O7pnRJImTpyoGTNmSJLy8vJ04sQJvfHGG9q9e/dt106YMEHz58/XhQsX+j3O7XbL7XY7HQ0AAIxCQ/6ckXg8nnIVoz+9vb06c+aMsrKyhnpaAAAwRji6MlJZWani4mI99NBD6u7uVnV1tZqamnT06FFJUllZmbKzsxUKhSRJr776qvLz8zVjxgx988032rp1qy5evKh169YN/zMBAACjkqMYuXTpksrKytTZ2Smfz6d58+bp6NGjevrppyVJ7e3tSkv7/mLL119/rfXr1yscDmvSpEnKy8tTc3PzgO4vAQAA9wZHMfLOO+/0+/OmpqaUx9u3b9f27dsdDwUAAO4dfDcNAAAwRYwAAABTxAgAADBFjAAAAFPECAAAMEWMAAAAU8QIAAAwRYwAAABTxAgAADBFjAAAAFPECAAAMEWMAAAAU8QIAAAwRYwAAABTxAgAADBFjAAAAFPECAAAMEWMAAAAU8QIAAAwRYwAAABTxAgAADBFjAAAAFPECAAAMEWMAAAAU8QIAAAwRYwAAABTxAgAADBFjAAAAFPECAAAMEWMAAAAU8QIAAAw5ShGdu3apXnz5snr9crr9SoYDKq+vr7fNQcOHNCsWbPk8Xg0d+5cHT58eEgDAwCAscVRjEybNk2vv/66Tp48qdbWVv34xz/WihUrdO7cuVse39zcrFWrVmnt2rU6ffq0SkpKVFJSorNnzw7L8AAAYPRzJRKJxFB+weTJk7V161atXbv2pp+Vlpaqp6dHhw4dSu7Lz89Xbm6uqqqqBnyOaDQqn8+nSCQir9c7lHFNXL16VcXFxZKk7gXPS+MmGE8EABixem8o49S7kqT6+nqlp6cbDzR4A339HvQ9I729vaqpqVFPT4+CweAtj2lpaVFhYWHKvqKiIrW0tPT7u2OxmKLRaMoGAADGJscxcubMGd1///1yu9164YUXVFtbq9mzZ9/y2HA4rMzMzJR9mZmZCofD/Z4jFArJ5/Mlt0Ag4HRMAAAwSjiOkZkzZ6qtrU1///vf9fOf/1yrV6/WP/7xj2EdqrKyUpFIJLl1dHQM6+8HAAAjx3inCyZOnKgZM2ZIkvLy8nTixAm98cYb2r17903H+v1+dXV1pezr6uqS3+/v9xxut1tut9vpaAAAYBQa8ueMxONxxWKxW/4sGAyqsbExZV9DQ0Of95gAAIB7j6MrI5WVlSouLtZDDz2k7u5uVVdXq6mpSUePHpUklZWVKTs7W6FQSJK0ceNGLV26VNu2bdPy5ctVU1Oj1tZW7dmzZ/ifCQAAGJUcxcilS5dUVlamzs5O+Xw+zZs3T0ePHtXTTz8tSWpvb1da2vcXWwoKClRdXa1XXnlFL7/8sh599FHV1dVpzpw5w/ssAADAqOUoRt55551+f97U1HTTvpUrV2rlypWOhgIAAPcOvpsGAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYchQjoVBIixYtUkZGhqZMmaKSkhKdP3++3zX79u2Ty+VK2Twez5CGBgAAY4ejGDl27JjKy8v1ySefqKGhQTdu3NAzzzyjnp6eftd5vV51dnYmt4sXLw5paAAAMHaMd3LwkSNHUh7v27dPU6ZM0cmTJ/Xkk0/2uc7lcsnv9w9uQgAAMKYN6Z6RSCQiSZo8eXK/x125ckUPP/ywAoGAVqxYoXPnzvV7fCwWUzQaTdkAAMDYNOgYicfj2rRpk5544gnNmTOnz+NmzpypvXv36uDBg3rvvfcUj8dVUFCgzz//vM81oVBIPp8vuQUCgcGOCQAARrhBx0h5ebnOnj2rmpqafo8LBoMqKytTbm6uli5dqg8//FAPPvigdu/e3eeayspKRSKR5NbR0THYMQEAwAjn6J6R72zYsEGHDh3S8ePHNW3aNEdrJ0yYoPnz5+vChQt9HuN2u+V2uwczGgAAGGUcXRlJJBLasGGDamtr9dFHH2n69OmOT9jb26szZ84oKyvL8VoAADD2OLoyUl5erurqah08eFAZGRkKh8OSJJ/Pp/T0dElSWVmZsrOzFQqFJEmvvvqq8vPzNWPGDH3zzTfaunWrLl68qHXr1g3zUwEAAKORoxjZtWuXJGnZsmUp+3//+9/rZz/7mSSpvb1daWnfX3D5+uuvtX79eoXDYU2aNEl5eXlqbm7W7NmzhzY5AAAYExzFSCKRuO0xTU1NKY+3b9+u7du3OxoKAADcO/huGgAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYMpRjIRCIS1atEgZGRmaMmWKSkpKdP78+duuO3DggGbNmiWPx6O5c+fq8OHDgx4YAACMLY5i5NixYyovL9cnn3yihoYG3bhxQ88884x6enr6XNPc3KxVq1Zp7dq1On36tEpKSlRSUqKzZ88OeXgAADD6uRKJRGKwiy9fvqwpU6bo2LFjevLJJ295TGlpqXp6enTo0KHkvvz8fOXm5qqqqmpA54lGo/L5fIpEIvJ6vYMd18zVq1dVXFwsSepe8Lw0boLxRACAEav3hjJOvStJqq+vV3p6uvFAgzfQ1+8h3TMSiUQkSZMnT+7zmJaWFhUWFqbsKyoqUktLS59rYrGYotFoygYAAMamQcdIPB7Xpk2b9MQTT2jOnDl9HhcOh5WZmZmyLzMzU+FwuM81oVBIPp8vuQUCgcGOCQAARrhBx0h5ebnOnj2rmpqa4ZxHklRZWalIJJLcOjo6hv0cAABgZBg/mEUbNmzQoUOHdPz4cU2bNq3fY/1+v7q6ulL2dXV1ye/397nG7XbL7XYPZjQAADDKOLoykkgktGHDBtXW1uqjjz7S9OnTb7smGAyqsbExZV9DQ4OCwaCzSQEAwJjk6MpIeXm5qqurdfDgQWVkZCTv+/D5fMm7fcvKypSdna1QKCRJ2rhxo5YuXapt27Zp+fLlqqmpUWtrq/bs2TPMTwUAAIxGjq6M7Nq1S5FIRMuWLVNWVlZy279/f/KY9vZ2dXZ2Jh8XFBSourpae/bsUU5Ojt5//33V1dX1e9MrAAC4dzi6MjKQjyRpamq6ad/KlSu1cuVKJ6cCAAD3CL6bBgAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmHIcI8ePH9ezzz6rqVOnyuVyqa6urt/jm5qa5HK5btrC4fBgZwYAAGOI4xjp6elRTk6Odu7c6Wjd+fPn1dnZmdymTJni9NQAAGAMGu90QXFxsYqLix2faMqUKfrhD3/oeB0AABjb7to9I7m5ucrKytLTTz+tv/3tb/0eG4vFFI1GUzYAADA23fEYycrKUlVVlT744AN98MEHCgQCWrZsmU6dOtXnmlAoJJ/Pl9wCgcCdHhMAABhx/DaNUzNnztTMmTOTjwsKCvSvf/1L27dv17vvvnvLNZWVlaqoqEg+jkajBAkAAGPUHY+RW1m8eLE+/vjjPn/udrvldrvv4kQAAMCKyeeMtLW1KSsry+LUAABghHF8ZeTKlSu6cOFC8vFnn32mtrY2TZ48WQ899JAqKyv1xRdf6I9//KMkaceOHZo+fboef/xxXbt2TW+//bY++ugj/eUvfxm+ZwEAAEYtxzHS2tqqp556Kvn4u3s7Vq9erX379qmzs1Pt7e3Jn1+/fl2//OUv9cUXX+i+++7TvHnz9Ne//jXldwAAgHuXK5FIJKyHuJ1oNCqfz6dIJCKv12s9jmNXr15NfjZL94LnpXETjCcCAIxYvTeUcerbP/Cor69Xenq68UCDN9DXb76bBgAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmHIcI8ePH9ezzz6rqVOnyuVyqa6u7rZrmpqatGDBArndbs2YMUP79u0bxKgAAGAschwjPT09ysnJ0c6dOwd0/Geffably5frqaeeUltbmzZt2qR169bp6NGjjocFAABjz3inC4qLi1VcXDzg46uqqjR9+nRt27ZNkvTYY4/p448/1vbt21VUVOT09KOeK/4fJayHAEaKREKK/+fb/04bL7lctvMAI4Dru38T9xDHMeJUS0uLCgsLU/YVFRVp06ZNfa6JxWKKxWLJx9Fo9E6Nd9fd3/Y/1iMAADCi3PEbWMPhsDIzM1P2ZWZmKhqN6urVq7dcEwqF5PP5klsgELjTYwIAACN3/MrIYFRWVqqioiL5OBqNjuog8Xg8qq+vtx4DGHGuXbum5557TpJUW1srj8djPBEwstwr/ybueIz4/X51dXWl7Ovq6pLX61V6evot17jdbrnd7js92l3jcrn6fK4AvuXxePh3Atyj7vjbNMFgUI2NjSn7GhoaFAwG7/SpAQDAKOA4Rq5cuaK2tja1tbVJ+vZPd9va2tTe3i7p27dYysrKkse/8MIL+ve//61f/epX+uc//6m33npLf/rTn/SLX/xieJ4BAAAY1RzHSGtrq+bPn6/58+dLkioqKjR//nxt3rxZktTZ2ZkME0maPn26/vznP6uhoUE5OTnatm2b3n777Xvyz3oBAMDNHN8zsmzZMiUSfX9Sxq0+XXXZsmU6ffq001MBAIB7AN9NAwAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAADAFDECAABMESMAAMAUMQIAAEwRIwAAwBQxAgAATA0qRnbu3KlHHnlEHo9HS5Ys0aefftrnsfv27ZPL5UrZPB7PoAcGAABji+MY2b9/vyoqKrRlyxadOnVKOTk5Kioq0qVLl/pc4/V61dnZmdwuXrw4pKEBAMDY4ThGfve732n9+vVas2aNZs+eraqqKt13333au3dvn2tcLpf8fn9yy8zMHNLQAABg7HAUI9evX9fJkydVWFj4/S9IS1NhYaFaWlr6XHflyhU9/PDDCgQCWrFihc6dO9fveWKxmKLRaMoGAADGJkcx8tVXX6m3t/emKxuZmZkKh8O3XDNz5kzt3btXBw8e1Hvvvad4PK6CggJ9/vnnfZ4nFArJ5/Mlt0Ag4GRMAAAwitzxv6YJBoMqKytTbm6uli5dqg8//FAPPvigdu/e3eeayspKRSKR5NbR0XGnxwQAAEbGOzn4gQce0Lhx49TV1ZWyv6urS36/f0C/Y8KECZo/f74uXLjQ5zFut1tut9vJaAAAYJRydGVk4sSJysvLU2NjY3JfPB5XY2OjgsHggH5Hb2+vzpw5o6ysLGeTAgCAMcnRlRFJqqio0OrVq7Vw4UItXrxYO3bsUE9Pj9asWSNJKisrU3Z2tkKhkCTp1VdfVX5+vmbMmKFvvvlGW7du1cWLF7Vu3brhfSYAAGBUchwjpaWlunz5sjZv3qxwOKzc3FwdOXIkeVNre3u70tK+v+Dy9ddfa/369QqHw5o0aZLy8vLU3Nys2bNnD9+zAAAAo5YrkUgkrIe4nWg0Kp/Pp0gkIq/Xaz0OgGFy9epVFRcXS5Lq6+uVnp5uPBGA4TTQ12++mwYAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJgaVIzs3LlTjzzyiDwej5YsWaJPP/203+MPHDigWbNmyePxaO7cuTp8+PCghgUAAGOP4xjZv3+/KioqtGXLFp06dUo5OTkqKirSpUuXbnl8c3OzVq1apbVr1+r06dMqKSlRSUmJzp49O+ThAQDA6OdKJBIJJwuWLFmiRYsW6c0335QkxeNxBQIBvfjii3rppZduOr60tFQ9PT06dOhQcl9+fr5yc3NVVVU1oHNGo1H5fD5FIhF5vV4n4wI3SSQSunbtmvUYkHTt2jU999xzkqTa2lp5PB7jiSBJHo9HLpfLegyMAQN9/R7v5Jdev35dJ0+eVGVlZXJfWlqaCgsL1dLScss1LS0tqqioSNlXVFSkurq6Ps8Ti8UUi8WSj6PRqJMxgX5du3ZNxcXF1mPg//kuSmCvvr5e6enp1mPgHuLobZqvvvpKvb29yszMTNmfmZmpcDh8yzXhcNjR8ZIUCoXk8/mSWyAQcDImAAAYRRxdGblbKisrU66mRKNRggTDxuPxqL6+3noM6Nu3zL67Cup2u3lrYITg7TLcbY5i5IEHHtC4cePU1dWVsr+rq0t+v/+Wa/x+v6PjpW//T8ntdjsZDRgwl8vFJegR5L777rMeAYAxR2/TTJw4UXl5eWpsbEzui8fjamxsVDAYvOWaYDCYcrwkNTQ09Hk8AAC4tzh+m6aiokKrV6/WwoULtXjxYu3YsUM9PT1as2aNJKmsrEzZ2dkKhUKSpI0bN2rp0qXatm2bli9frpqaGrW2tmrPnj3D+0wAAMCo5DhGSktLdfnyZW3evFnhcFi5ubk6cuRI8ibV9vZ2paV9f8GloKBA1dXVeuWVV/Tyyy/r0UcfVV1dnebMmTN8zwIAAIxajj9nxAKfMwIAwOgz0NdvvpsGAACYIkYAAIApYgQAAJgiRgAAgCliBAAAmCJGAACAKWIEAACYIkYAAIApYgQAAJhy/HHwFr77kNhoNGo8CQAAGKjvXrdv92HvoyJGuru7JUmBQMB4EgAA4FR3d7d8Pl+fPx8V300Tj8f15ZdfKiMjQy6Xy3ocAMMoGo0qEAioo6OD754CxphEIqHu7m5NnTo15Ut0/79RESMAxi6+CBMAN7ACAABTxAgAADBFjAAw5Xa7tWXLFrndbutRABjhnhEAAGCKKyMAAMAUMQIAAEwRIwAAwBQxAgAATBEjAMzs3LlTjzzyiDwej5YsWaJPP/3UeiQABogRACb279+viooKbdmyRadOnVJOTo6Kiop06dIl69EA3GX8aS8AE0uWLNGiRYv05ptvSvr2O6gCgYBefPFFvfTSS8bTAbibuDIC4K67fv26Tp48qcLCwuS+tLQ0FRYWqqWlxXAyABaIEQB33VdffaXe3l5lZmam7M/MzFQ4HDaaCoAVYgQAAJgiRgDcdQ888IDGjRunrq6ulP1dXV3y+/1GUwGwQowAuOsmTpyovLw8NTY2JvfF43E1NjYqGAwaTgbAwnjrAQDcmyoqKrR69WotXLhQixcv1o4dO9TT06M1a9ZYjwbgLiNGAJgoLS3V5cuXtXnzZoXDYeXm5urIkSM33dQKYOzjc0YAAIAp7hkBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABgihgBAACmiBEAAGCKGAEAAKaIEQAAYIoYAQAApogRAABg6n8BnfO3/3mpVjkAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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Y+3atRER8dRTT8XSpUuju7s7Vq5c+ap95PP54hXwERG5XC4aGhpiaGgoamtrxzoaAJBQLpeLbDZ7VK/f47rmY2hoKCIi5s6dGxERPT09cfjw4Whqaipus2TJkli0aFF0d3e/5j42bdoU2Wy2uDQ0NIxnJABgkhtzfIyMjMSGDRvi/PPPj7POOisiIgYGBmLmzJkxZ86cUdvW1dXFwMDAa+6nvb09hoaGiktfX99YRwIApoAx3+ejtbU1nnzyyXj44YfHNUB1dXVUV1ePax8AwNQxpiMf119/fdx3333xm9/8Jk499dTi+vr6+jh06FDs379/1PaDg4NRX18/rkEBgOmhpPgoFApx/fXXx65du+LXv/51LF68eNTjy5cvjxkzZkRnZ2dx3b59+6K3tzcaGxuPzcQAwJRW0mmX1tbW2L59e/z85z+P2bNnF6/jyGazUVNTE9lsNtatWxdtbW0xd+7cqK2tjfXr10djY+NrvtMFADj+lPRW29f7IKht27bF1VdfHREv32TshhtuiHvuuSfy+XysWrUqtm7detSnXUp5qw4AMDmU8vo9rvt8TATxAQBTTymv35PuU21faaFcLlfmSQCAo/XK6/bRHNOYdPFx4MCBiAg3GwOAKejAgQORzWbfcJtJd9plZGQk+vv7Y/bs2a97jQkwNb3y8Ql9fX1Oq8I0UygU4sCBA7Fw4cKorHzjN9NOuvgApi/XdAER4/xsFwCAUokPACAp8QEkU11dHTfddJPPc4LjnGs+AICkHPkAAJISHwBAUuIDAEhKfAAASYkPACAp8QEks2XLljj99NMjk8nEihUr4tFHHy33SEAZiA8giR//+MfR1tYWN910Uzz++OPxzne+M1atWhXPP/98uUcDEnOfDyCJFStWxHnnnRff+c53IuLlD5FsaGiI9evXx+c///kyTwek5MgHMOEOHToUPT090dTUVFxXWVkZTU1N0d3dXcbJgHIQH8CE+8c//hFHjhyJurq6Uevr6upiYGCgTFMB5SI+AICkxAcw4d70pjfFCSecEIODg6PWDw4ORn19fZmmAspFfAATbubMmbF8+fLo7OwsrhsZGYnOzs5obGws42RAOVSVewDg+NDW1hZXXXVVnHvuufHud787br311jh48GBcc8015R4NSEx8AElcfvnl8cILL8SNN94YAwMD8a53vSvuv//+V12ECkx/7vMBACTlmg8AICnxAQAkJT4AgKTEBwCQlPgAAJISHwBAUuIDAEhKfAAASYkPACAp8QEAJCU+AICk/h/I8NA1VT1uBAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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6cCBA1m8eHHmzJlzRN9AAgCObnX/GnWSrF69OkuXLs1LL72UyZMnZ8mSJVmwYEFtf7VazY033ph77703u3btyvnnn5+77747v/iLv1hbs3PnzixevDiPPfZYmpubM3v27KxcuTInnHDCEc3ga9QAUJ4jff8ekoAZDgQMAJSnYdeBAQAYagIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4ggYAKA4AgYAKI6AAQCKI2AAgOIIGACgOAIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4ggYAKA4AgYAKI6AAQCKI2AAgOIIGACgOAIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4ggYAKA4AgYAKI6AAQCKI2AAgOIIGACgOAIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4ggYAKA4AgYAKI6AAQCKI2AAgOIIGACgOAIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4gx5wNx2221pamrKVVddVdu2d+/eLFq0KCeddFJOOOGEzJ49O319fYPut3Xr1syaNSvHH398Jk6cmOuuuy4HDx4c6nEBgAIMacA888wz+du//dv8yq/8yqDtV199dR577LE88sgjefLJJ7N9+/Z85jOfqe0/dOhQZs2alf379+epp57KAw88kPvvvz/Lli0bynEBgEIMWcDs3r07c+fOzde+9rWceOKJte39/f257777cvvtt+fXf/3Xc9ZZZ+XrX/96nnrqqXz/+99PkvzTP/1TfvSjH+Ub3/hGpk2blosvvji33HJL7rrrruzfv3+oRgYACjFkAbNo0aLMmjUr3d3dg7Zv3LgxBw4cGLT9tNNOyymnnJKenp4kSU9PT84444y0t7fX1sycOTOVSiWbNm067PPt27cvlUpl0A0AODqNHIoH/eY3v5lnn302zzzzzDv29fb2ZtSoURk/fvyg7e3t7ent7a2teWu8vLn/zX2Hs3z58tx00011mB4AGO7qfgRm27Zt+eM//uM8+OCDGT16dL0f/l0tXbo0/f39tdu2bds+sOcGAD5YdQ+YjRs3ZseOHfn4xz+ekSNHZuTIkXnyySezcuXKjBw5Mu3t7dm/f3927do16H59fX3p6OhIknR0dLzjW0lv/vvNNW/X0tKS1tbWQTcA4OhU94C54IIL8vzzz+e5556r3aZPn565c+fW/j7uuOOybt262n22bNmSrVu3pqurK0nS1dWV559/Pjt27KitWbt2bVpbWzN16tR6jwwAFKbu58CMGzcup59++qBtY8eOzUknnVTbPn/+/CxZsiQTJkxIa2trrrzyynR1deWcc85Jklx44YWZOnVqPv/5z2fFihXp7e3NF77whSxatCgtLS31HhkAKMyQnMT7Xv76r/86zc3NmT17dvbt25eZM2fm7rvvru0fMWJEVq9enSuuuCJdXV0ZO3Zs5s2bl5tvvrkR4wIAw0xTtVqtNnqIoVCpVNLW1pb+/n7nwwBAIY70/dtvIQEAxREwAEBxBAwAUBwBAwAUR8AAAMURMABAcQQMAFAcAQMAFEfAAADFETAAQHEEDABQHAEDABRHwAAAxREwAEBxBAwAUBwBAwAUR8AAAMURMABAcQQMAFAcAQMAFEfAAADFETAAQHEEDABQHAEDABRHwAAAxREwAEBxBAwAUBwBAwAUR8AAAMURMABAcQQMAFAcAQMAFEfAAADFETAAQHEEDABQHAEDABRHwAAAxREwAEBxBAwAUBwBAwAUR8AAAMURMABAcQQMAFAcAQMAFEfAAADFETAAQHEEDABQHAEDABRHwAAAxREwAEBxBAwAUBwBAwAUR8AAAMURMABAcQQMAFAcAQMAFKfuAbN8+fJ84hOfyLhx4zJx4sRceuml2bJly6A1e/fuzaJFi3LSSSflhBNOyOzZs9PX1zdozdatWzNr1qwcf/zxmThxYq677rocPHiw3uMCAAWqe8A8+eSTWbRoUb7//e9n7dq1OXDgQC688MLs2bOntubqq6/OY489lkceeSRPPvlktm/fns985jO1/YcOHcqsWbOyf//+PPXUU3nggQdy//33Z9myZfUeFwAoUFO1Wq0O5RO8+uqrmThxYp588sl88pOfTH9/f37u534uDz30UD772c8mSV544YVMmTIlPT09Oeecc/L444/nU5/6VLZv35729vYkyapVq/Knf/qnefXVVzNq1Kj3fN5KpZK2trb09/entbV1KF8iAFAnR/r+PeTnwPT39ydJJkyYkCTZuHFjDhw4kO7u7tqa0047Laecckp6enqSJD09PTnjjDNq8ZIkM2fOTKVSyaZNmw77PPv27UulUhl0AwCOTkMaMAMDA7nqqqty3nnn5fTTT0+S9Pb2ZtSoURk/fvygte3t7ent7a2teWu8vLn/zX2Hs3z58rS1tdVukyZNqvOrAQCGiyENmEWLFuU//uM/8s1vfnMonyZJsnTp0vT399du27ZtG/LnBAAaY+RQPfDixYuzevXqrF+/Ph/+8Idr2zs6OrJ///7s2rVr0FGYvr6+dHR01NY8/fTTgx7vzW8pvbnm7VpaWtLS0lLnVwEADEd1PwJTrVazePHiPProo3niiScyefLkQfvPOuusHHfccVm3bl1t25YtW7J169Z0dXUlSbq6uvL8889nx44dtTVr165Na2trpk6dWu+RAYDC1P0IzKJFi/LQQw/lH/7hHzJu3LjaOSttbW0ZM2ZM2traMn/+/CxZsiQTJkxIa2trrrzyynR1deWcc85Jklx44YWZOnVqPv/5z2fFihXp7e3NF77whSxatMhRFgCg/l+jbmpqOuz2r3/96/m93/u9JG9cyO6aa67J3//932ffvn2ZOXNm7r777kEfD/34xz/OFVdcke9+97sZO3Zs5s2bl9tuuy0jRx5Zc/kaNQCU50jfv4f8OjCNImAAoDzD5jowAAD1JmAAgOIIGACgOAIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4ggYAKA4AgYAKI6AAQCKI2AAgOIIGACgOAIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4ggYAKA4AgYAKI6AAQCKI2AAgOIIGACgOAIGACiOgAEAiiNgAIDiCBgAoDgCBgAojoABAIojYACA4ggYAKA4AgYAKI6AAQCKI2AAgOIIGACgOAIGACiOgAEAiiNgAIDiCBigOE899VQuu+yyPPXUU40eBWgQAQMUZe/evbnlllvS19eXW265JXv37m30SEADCBigKPfdd19ef/31JMnrr7+ev/u7v2vwREAjCBigGK+88koeeeSRQdu+9a1v5ZVXXmnQRECjCBigCNVqNTfffPNh9918882pVqsf8ERAI41s9ABQgmq16lyLBvuv//qvvPjii4fd9+KLL+aFF17IRz/60Q92KGpGjx6dpqamRo/BMUTAwBHYu3dvLr744kaPwf/HFVdc0egRjmmPP/54xowZ0+gxOIb4CAkAKI4jMHAERo8enccff7zRYxzzfvCDH+TP/uzP3rF9+fLlmTZt2gc/EDWjR49u9AgcYwQMHIGmpiaHx4eBc889N1OmTMnmzZtr204//fR0dXU1cCqgEXyEBBTlxhtvrP3d1NSUW2+9tYHTAI3iCMww5VsvcHgtLS21vz/72c+mpaWldmE74Nj5RlhT9Si9eEKlUklbW1v6+/vT2tra6HHet9dff923XgB430r/RtiRvn/7CAkAKM6w/gjprrvuyle+8pX09vbmzDPPzJ133pmzzz670WN9IN56YGz3GZ9LtXlEA6eBYaSaZODgG383j0yO/iPl8J6aBg7lhOff+JmNo/SDlXcYtgHz8MMPZ8mSJVm1alVmzJiRO+64IzNnzsyWLVsyceLERo835Pbt21f7+83/UQLAe9m3b1+OP/74Ro8x5IbtR0i33357FixYkMsvvzxTp07NqlWrcvzxx/vlWQBgeAbM/v37s3HjxnR3d9e2NTc3p7u7Oz09PQ2c7IPz1m9aAMCROlbeP4blR0g/+clPcujQobS3tw/a3t7enhdeeOGw99m3b9+gj10qlcqQzjjUxowZ48qvw8jevXvzG7/xG40eA4atRx991NV4h4lj5b+HYRkwP4vly5fnpptuavQYdePKr8OLnxIYPqrVau3/rLS0tBwT17sowbFy7RGGj2EZMB/60IcyYsSI9PX1Ddre19eXjo6Ow95n6dKlWbJkSe3flUolkyZNGtI5OXYIyuHlWDhBEfj/G5bnwIwaNSpnnXVW1q1bV9s2MDCQdevWvetvnrS0tKS1tXXQDQA4Og3LIzBJsmTJksybNy/Tp0/P2WefnTvuuCN79uzJ5Zdf3ujRAIAGG7YBc9lll+XVV1/NsmXL0tvbm2nTpmXNmjXvOLEXADj2+C0kAGDY8FtIAMBRS8AAAMURMABAcQQMAFAcAQMAFEfAAADFETAAQHEEDABQHAEDABRn2P6UwP/VmxcYrlQqDZ4EADhSb75vv9cPBRy1AfPaa68lSSZNmtTgSQCA9+u1115LW1vbu+4/an8LaWBgINu3b8+4cePS1NTU6HGAOqpUKpk0aVK2bdvmt87gKFOtVvPaa6+ls7Mzzc3vfqbLURswwNHLj7UCTuIFAIojYACA4ggYoDgtLS258cYb09LS0uhRgAZxDgwAUBxHYACA4ggYAKA4AgYAKI6AAQCKI2CAotx111356Ec/mtGjR2fGjBl5+umnGz0S0AACBijGww8/nCVLluTGG2/Ms88+mzPPPDMzZ87Mjh07Gj0a8AHzNWqgGDNmzMgnPvGJfPWrX03yxm+eTZo0KVdeeWWuv/76Bk8HfJAcgQGKsH///mzcuDHd3d21bc3Nzenu7k5PT08DJwMaQcAARfjJT36SQ4cOpb29fdD29vb29Pb2NmgqoFEEDABQHAEDFOFDH/pQRowYkb6+vkHb+/r60tHR0aCpgEYRMEARRo0albPOOivr1q2rbRsYGMi6devS1dXVwMmARhjZ6AEAjtSSJUsyb968TJ8+PWeffXbuuOOO7NmzJ5dffnmjRwM+YAIGKMZll12WV199NcuWLUtvb2+mTZuWNWvWvOPEXuDo5zowAEBxnAMDABRHwAAAxREwAEBxBAwAUBwBAwAUR8AAAMURMABAcQQMAFAcAQMAFEfAAADFETAAQHEEDABQnP8HPRRLi1tGDB8AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Detection of Outlier in Family_Members variable"
      ],
      "metadata": {
        "id": "MhgCWARLeQug"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.boxplot(data=X1,y='Family_Members')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 428
        },
        "id": "0FYQZ91ceu0b",
        "outputId": "3db1c394-a87d-43ef-e031-aafff112dc8b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: ylabel='Family_Members'>"
            ]
          },
          "metadata": {},
          "execution_count": 105
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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MArlBgAAGIVyAwAAjEK5AQAARqHcAAAAo1BuAACAUSg3AADAKJQbAABgFMoNAAAwCuUGAAAYJeJyU19frxMnToQeHz58WIsXL9Zrr70WlWAAAACRiLjcTJ06VatXr5YkHTt2TFdddZXy8vI0depULVu2LGoBAQAAwhFxuSkpKdE111wjScrPz1diYqIOHz6s1atXa8mSJVELCAAAEI6Iy82JEycUHx8vSXrttdf09a9/XV26dNGYMWN0+PDhqAUEAAAIR8TlJjU1VZs2bdKRI0dUWFiom266SZJUXV0tl8sVtYAAAADhiLjcPPTQQ/rxj3+sSy+9VFdddZXS09MlnZzFGTVqVNQCAgAAhOPCSA/85je/qXHjxqmiokIjRowI7b/hhhs0bdq0qIQDAAAIV0TlprGxUXFxcSotLT1tluYrX/lKVIIBAABEIqLTUl27dlWfPn3U1NQU7TwAAADtEvGam5/+9Kf6yU9+ok8//TSaeQAAANol4jU3S5cu1cGDB5WcnKy+ffuqe/fuzZ4vKSlpdzgAAIBwRVxuMjMzoxgDAAAgOiIuNw8//HA0cwAAAERFu+4KfuzYMa1cuVK5ubmhtTclJSX65JNPohIOAAAgXBHP3Lz77ruaMGGC3G63PvroI82ePVs9e/bUhg0bVF5eHrqpJgAAQEeKeOZm/vz5uv322/WPf/xDTqcztH/SpEnavn17VMIBAACEK+Jys2fPHn3ve987bX9KSooqKyvbFQoAACBSEZcbh8Mhn8932v6///3v+tKXvtSuUAAAAJGKuNx87Wtf04IFC9TY2ChJsixL5eXleuCBB/SNb3wjagEBAADCEXG5ycvLU21trRISElRfX6/rrrtOqampio+P1xNPPBHNjAAAAG0W8dVSbrdbRUVF2rFjh959913V1tYqLS1NEyZMiGY+AACAsERcbk4ZN26cxo0bF40sAAAA7dauH/HbunWrpkyZov79+6t///6aMmWKXn/99WhlAwAACFvE5ebXv/61br75ZsXHx2vevHmaN2+eXC6XJk2apGeeeSaaGQEAANos4tNSCxcu1KJFizR37tzQvnvuuUdXX321Fi5cqDlz5kQlIAAAQDginrk5duyYbr755tP233TTTaqpqWlXKAAAgEi163duNm7ceNr+zZs3a8qUKW16jWXLlmn48OFyuVxyuVxKT09XQUFBq8esX79eAwcOlNPp1LBhw7Rly5aI8gMAADOFdVpqyZIloT8PHjxYTzzxhIqLi5Weni5J2r17t9566y396Ec/atPr9e7dW08++aQuv/xyBYNB/e53v9PUqVO1f/9+DRky5LTxO3fuVFZWlrxer6ZMmaI1a9YoMzNTJSUlGjp0aDgfBQAAGMoKBoPBtg7u169f217UsvThhx9GFKhnz5566qmndOedd5723G233aa6ujq98soroX1jxozRyJEj9eyzz7bp9X0+n9xut2pqauRyuSLKaLdgMKiGhga7YwAxpaGhQdOmTZMkbdy4sdkNfQGc5HQ6ZVmW3TEiEs73d1gzN4cOHWpXsNY0NTVp/fr1qqurC80E/a9du3Zp/vz5zfZNnDhRmzZtOuPr+v1++f3+0OOW7ofV2TQ0NCgjI8PuGEDMOlVyADRXUFCguLg4u2Occ+36nZtoKCsr08UXXyyHw6G7775bGzdu1ODBg1scW1lZqcTExGb7EhMTW70LudfrldvtDm0ejyeq+QEAQGyJ+FLwYDCo/Px8vfnmm6qurlYgEGj2/IYNG9r0OgMGDFBpaalqamqUn5+vWbNmadu2bWcsOOHKzc1tNtvj8/mMKji1I7MU7NLuH5oGOr9gUAr8++Sfu1woddKpdyDarMC/dXHpWrtjdKiIvxVzcnK0fPlyXX/99UpMTIz4HN5FF12k1NRUSdLo0aO1Z88ePf3001q+fPlpY5OSklRVVdVsX1VVlZKSks74+g6HQw6HI6JsnUGwy4XSBV3tjgHEiIvsDgDEnDYvrDVIxOXmhRde0IYNGzRp0qRo5lEgEGi2Rua/paena+vWrcrJyQntKyoqOuMaHQAAcP5p113BL7vssna9eW5urjIyMtSnTx8dP35ca9asUXFxsQoLCyVJ2dnZSklJkdfrlSTNmzdP1113nfLy8jR58mStW7dOe/fu1YoVK9qVAwAAmCPiBcWPPPKIHn30UdXX10f85tXV1crOztaAAQN0ww03aM+ePSosLNSNN94oSSovL1dFRUVo/NixY7VmzRqtWLFCI0aMUH5+vjZt2sRv3AAAgJCIZ25uvfVWrV27VgkJCbr00kvVtWvzdR8lJSVnfY3f/OY3rT5fXFx82r7p06dr+vTpYWUFAADnj4jLzaxZs7Rv3z7NnDmzXQuKAQAAoinicvPqq6+qsLBQ48aNi2YeAACAdol4zY3H4+m0ty8AAADmirjc5OXl6f7779dHH30UxTgAAADtE/FpqZkzZ+rEiRPq37+/unXrdtqC4k8//bTd4QAAAMIVcblZvHhxFGMAAABER7uulgIAAIg17bor+AcffKCf/exnysrKUnV1taSTt1N///33oxIOAAAgXBGXm23btmnYsGF6++23tWHDBtXW1kqS3nnnHT388MNRCwgAABCOiMvNgw8+qMcff1xFRUW66KL/3In3q1/9qnbv3h2VcAAAAOGKuNyUlZVp2rRpp+1PSEjQv/71r3aFAgAAiFTE5eaSSy5pdlPLU/bv36+UlJR2hQIAAIhUxOXmW9/6lh544AFVVlbKsiwFAgG99dZb+vGPf6zs7OxoZgQAAGiziMvNwoULNXDgQHk8HtXW1mrw4MG69tprNXbsWP3sZz+LZkYAAIA2i/h3bi666CI999xz+r//+z+99957qq2t1ahRo3T55ZdHMx8AAEBYIi43p/Tp00d9+vSJRhYAAIB2C7vcLFiwoE3jHnroobDDAAAAtFfY5eaRRx5RcnKyEhISFAwGWxxjWRblBgAA2CLscpORkaE33nhDV1xxhb7zne9oypQp6tKlXXdxAAAAiJqwW8mrr76qDz74QFdddZXuu+8+paSk6IEHHtCBAwfORT4AAICwRDTlkpycrNzcXB04cEB/+MMfVF1drSuvvFJXX3216uvro50RAACgzdp9tdSVV16pjz76SH/961+1f/9+NTY2Ki4uLhrZAAAAwhbxYpldu3Zp9uzZSkpK0q9+9SvNmjVLR48elcvlimY+AACAsIQ9c/Pzn/9cq1at0r/+9S/NmDFDf/nLXzR8+PBzkQ0AACBsYZebBx98UH369NGtt94qy7K0atWqFsf98pe/bG82AACAsIVdbq699lpZlqX333//jGMsy2pXKAAAgEiFXW6Ki4vPQQwAAIDoOOe/vudyufThhx+e67cBAACQ1AHl5ky3aAAAADgXuG8CAAAwCuUGAAAYhXIDAACMcs7LDZeFAwCAjsSCYgAAYJSIy82bb77ZpnEFBQVKSUmJ9G0AAADCEnG5ufnmm9W/f389/vjjOnLkyBnHjRs3Tg6HI9K3AQAACEvE5eaTTz7R3LlzlZ+fr8suu0wTJ07USy+9pM8//zya+QAAAMIScbn54he/qHvvvVelpaV6++239eUvf1k/+MEPlJycrHvuuUfvvPPOWV/D6/XqyiuvVHx8vBISEpSZmakDBw60esyqVatkWVazzel0RvoxAACAYaKyoDgtLU25ubmaO3euamtr9fzzz2v06NG65pprWr3B5rZt2zRnzhzt3r1bRUVFamxs1E033aS6urpW38/lcqmioiK0HT58OBofAwAAGKBd5aaxsVH5+fmaNGmS+vbtq8LCQi1dulRVVVU6ePCg+vbtq+nTp5/x+D//+c+6/fbbNWTIEI0YMUKrVq1SeXm59u3b1+r7WpalpKSk0JaYmNiejwEAAAwS9l3BT/nhD3+otWvXKhgM6tvf/rZ+/vOfa+jQoaHnu3fvrl/84hdKTk5u82vW1NRIknr27NnquNraWvXt21eBQEBpaWlauHChhgwZ0uJYv98vv98feuzz+dqcBwAAdD4Rz9z89a9/1a9+9SsdPXpUixcvblZsTvniF7/Y5kvGA4GAcnJydPXVV7f4WqcMGDBAzz//vDZv3qwXX3xRgUBAY8eO1ccff9zieK/XK7fbHdo8Hk/bPiAAAOiUrGCM/Mre97//fRUUFGjHjh3q3bt3m49rbGzUoEGDlJWVpccee+y051uaufF4PKqpqZHL5YpK9o5WX1+vjIwMSdLxtG9LF3S1OREAIGY1NSq+5AVJJ397Li4uzuZAkfH5fHK73W36/g7rtNQf//jHNo/92te+1uaxc+fO1SuvvKLt27eHVWwkqWvXrho1apQOHjzY4vMOh4Pf2QEA4DwSVrnJzMxs0zjLstTU1HTWccFgUD/84Q+1ceNGFRcXq1+/fuHEkSQ1NTWprKxMkyZNCvtYAABgnrDKTSAQiOqbz5kzR2vWrNHmzZsVHx+vyspKSZLb7Q5Nm2VnZyslJUVer1eStGDBAo0ZM0apqak6duyYnnrqKR0+fFh33XVXVLMBAIDOKeKrpaJh2bJlkqTx48c32//b3/5Wt99+uySpvLxcXbr8Z93zZ599ptmzZ6uyslI9evTQ6NGjtXPnTg0ePLijYgMAgBgWVrlZsmSJvvvd78rpdGrJkiWtjr3nnnvO+nptWctcXFzc7PGiRYu0aNGisx4HAADOT2GVm0WLFmnGjBlyOp2tFgzLstpUbgAAAKItrHJz6NChFv8MAAAQK6JybykAAIBYEfGC4mAwqPz8fL355puqrq4+7UqqDRs2tDscAABAuCIuNzk5OVq+fLmuv/56JSYmyrKsaOYCAACISMTl5oUXXtCGDRv48TwAABBTIl5z43a7ddlll0UzCwAAQLtFXG4eeeQRPfroo6qvr49mHgAAgHaJ+LTUrbfeqrVr1yohIUGXXnqpunZtfmfqkpKSdocDAAAIV8TlZtasWdq3b59mzpzJgmIAABAzIi43r776qgoLCzVu3Lho5gEAAGiXiNfceDweuVyuaGYBAABot4jLTV5enu6//3599NFHUYwDAADQPhGflpo5c6ZOnDih/v37q1u3bqctKP7000/bHQ4AACBcEZebxYsXRzEGAABAdLTraikAAIBYE3G5+W8NDQ36/PPPm+1jsTEAALBDxAuK6+rqNHfuXCUkJKh79+7q0aNHsw0AAMAOEZeb+++/X2+88YaWLVsmh8OhlStX6tFHH1VycrJWr14dzYwAAABtFvFpqT/96U9avXq1xo8frzvuuEPXXHONUlNT1bdvX/3+97/XjBkzopkTAACgTSKeufn0009DdwV3uVyhS7/HjRun7du3RycdAABAmCIuN5dddpkOHTokSRo4cKBeeuklSSdndC655JKohAMAAAhX2OXmww8/VCAQ0B133KF33nlHkvTggw/qmWeekdPp1L333qv77rsv6kEBAADaIuw1N5dffrkqKip07733SpJuu+02LVmyRH/729+0b98+paamavjw4VEPCgAA0BZhz9wEg8Fmj7ds2aK6ujr17dtXX//61yk2AADAVhGvuQEAAIhFYZcby7JkWdZp+wAAAGJB2GtugsGgbr/9djkcDkknb71w9913q3v37s3GbdiwIToJAQAAwhB2ufnfG2bOnDkzamEAAADaK+xy89vf/vZc5AAAAIgKFhQDAACjUG4AAIBRKDcAAMAolBsAAGAUyg0AADAK5QYAABiFcgMAAIxia7nxer268sorFR8fr4SEBGVmZurAgQNnPW79+vUaOHCgnE6nhg0bpi1btnRAWgAA0BnYWm62bdumOXPmaPfu3SoqKlJjY6Nuuukm1dXVnfGYnTt3KisrS3feeaf279+vzMxMZWZm6r333uvA5AAAIFZZwWAwaHeIU/75z38qISFB27Zt07XXXtvimNtuu011dXV65ZVXQvvGjBmjkSNH6tlnnz3re/h8PrndbtXU1MjlckUte0c6ceKEJk2aJEmqHTZd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Sk9PtykRAACINbaWm9raWpWWlqq0tFTSyUu9S0tLVV5eLunkKaXs7OzQ+Lvvvlsffvih7r//fv3tb3/Tr3/9a7300ku699577YgPAABikK3lZu/evRo1apRGjRolSZo/f75GjRqlhx56SJJUUVERKjqS1K9fP7366qsqKirSiBEjlJeXp5UrV3IZOAAACLGCwWDQ7hAdyefzye12q6amRi6Xy+44AACgDcL5/u5Ua24AAADOhnIDAACMQrkBAABGodwAAACjUG4AAIBRKDcAAMAolBsAAGAUyg0AADAK5QYAABilU90VPBpO/SCzz+ezOQkAAGirU9/bbbmxwnlXbo4fPy5J8ng8NicBAADhOn78uNxud6tjzrt7SwUCAR09elTx8fGyLMvuOACiyOfzyePx6MiRI9w7DjBMMBjU8ePHlZycrC5dWl9Vc96VGwDm4sa4ACQWFAMAAMNQbgAAgFEoNwCM4XA49PDDD8vhcNgdBYCNWHMDAACMwswNAAAwCuUGAAAYhXIDAACMQrkBAABGodwAAACjUG4AAIBRKDcAAMAolBsAAGCU/wfz7JM7kEz3VQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Q1=X1['Family_Members'].quantile(0.25)\n",
        "Q3=X1['Family_Members'].quantile(0.75)\n",
        "IQR = Q3 - Q1\n",
        "lower = Q1 - 1.5*IQR\n",
        "upper = Q3 + 1.5*IQR\n",
        "\n",
        "X1['Family_Members']=np.where(X1['Family_Members']<lower,lower,X1['Family_Members'])\n",
        "X1['Family_Members']=np.where(X1['Family_Members']>upper,upper,X1['Family_Members'])\n",
        "sns.boxplot(y=credit_card_raw['Family_Members'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 428
        },
        "id": "31ojPYZ8I6Ru",
        "outputId": "1be9aa23-cdfc-4e89-e9de-d78fd28ca105"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: ylabel='Family_Members'>"
            ]
          },
          "metadata": {},
          "execution_count": 72
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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y5Ytdeedd6qoqEhvv/22ysvLdfjwYUlSenq6LrroIqdiAwCABOLopdkbNmxQ3759z9o+atQozZo1S1lZWVW+b/369erTp0+tPoNLswEAMI8xl2b36dNHNXWpBLkFDgAASGBGXM0EAABQHcoMAAAwGmUGAAAYjTIDAACMRpkBAABGo8wAAACjUWYAAIDRKDMAAMBolBkAAGA0ygwAADAaZQYAABiNMgMAAIxGmQEAAEajzAAAAKNRZgAAgNEoMwAAwGiUGQAAYDTKDAAAMBplBgAAGI0yAwAAjEaZAQAARqPMAAAAo1FmAACA0SgzAADAaJQZAABgNMoMAAAwGmUGAAAYjTIDAACMRpkBAABGo8wAAACjUWYAAIDRKDMAAMBolBkAAGA0ygwAADAaZQYAABgt5jJz+vRpnTp1KrK+f/9+LVq0SO+9915cggEAANRGzGVm8ODBWr58uSTp+PHjuvbaa7VgwQINHjxYS5cujVtAAACAmsRcZoqKinT99ddLklavXq0WLVpo//79Wr58uRYvXlyrY2zcuFGDBg2Sz+eTZVlas2ZNpddt29bMmTPVsmVLpaamql+/ftq1a1eskQEAwHko5jJz6tQppaWlSZLee+893X777UpKStJPf/pT7d+/v1bHOHnypLp06aIlS5ZU+fozzzyjxYsX64UXXtCWLVvUuHFj3XzzzQqFQrHGBgAA55mYy8xll12mNWvWqLi4WGvXrtVNN90kSSopKZHH46nVMfr37685c+Zo6NChZ71m27YWLVqk6dOna/DgwercubOWL1+uQ4cOnTWCAwAALlwxl5mZM2fq0UcfVdu2bXXttdfquuuuk3RmlKZr1651DrZ3714dPnxY/fr1i2zzer269tpr9emnn1b7vnA4rGAwWGkBAADnr5RY33jnnXeqd+/e+vbbb9WlS5fI9htvvLHKkZZoHT58WJLUokWLSttbtGgRea0q8+fP1+zZs+v8+QAAwAwxjcyUlZUpJSVF3333nbp27aqkpP9/mB49eqhDhw5xCxitnJwcBQKByFJcXOxYFgAAUP9iKjONGjVSZmamysvL450n4tJLL5UkHTlypNL2I0eORF6risvlksfjqbQAAIDzV8xzZv73f/9Xjz/+uI4dOxbPPBFZWVm69NJL9f7770e2BYNBbdmyJTI/BwAAIOY5M88995x2794tn8+n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q6SkRD6fT8OGDdPMmTMrPdyyJpxmAgDAPNF8f0ddZp555hktW7ZM3333nYYPH67Ro0erc+fOdQpcnygzAACYp17LTFJSkjIzMzVw4MAaR0dquhdMQ6LMAABgnnq9NPuGG26QZVn68ssvq93HsqxoDwsAABCTqMvMhg0b6iEGAABAbOrnbnf/wePxaM+ePfX9MQAA4AJV72UmxoulAAAAaqXeywwAAEB9oswAAACjUWYAAIDR6r3McJk2AACoT0wABgAARou5zKxfv75W+xUUFKhVq1axfgwAAECNYi4zt9xyi9q3b685c+aouLi42v169+4tl8sV68cAAADUKOYyc/DgQY0bN06rV69Wu3btdPPNN2vVqlX617/+Fc98AAAANYq5zDRv3lyTJk3S9u3btWXLFl1xxRX6zW9+I5/Pp/Hjx2vHjh3xzAkAAFCluEwA7tatm3JycjRu3DiVlpbq1Vdf1U9+8hNdf/31NT6QEgAAoK7qVGbKysq0evVqDRgwQG3atNHatWv13HPP6ciRI9q9e7fatGmju+66K15ZAQAAzmLZMV47/fDDD2vlypWybVsjR47Ur371K3Xq1KnSPocPH5bP51NFRUVcwsYiGAzK6/UqEAjI4/E4lgMAANReNN/fKbF+yFdffaXf/e53uv3226u9Wql58+a1voQbAAAgFjGPzJiCkRkAAMxTbyMzb731Vq33ve2226I5NAAAQEyiKjNDhgyp1X6WZam8vDyWPAAAAFGJqsw4OZEXAACgKvX+oEkAAID6FNXIzOLFi/XAAw/I7XZr8eLFNe47fvz4OgUDAACojaiuZsrKytLWrVvVrFkzZWVlVX9Qy9KePXviErCuuJoJAADz1NvVTHv37q3yZwAAAKcwZwYAABgt5jsA27at1atXa/369SopKTnrSqe8vLw6hwMAADiXmMvMxIkT9eKLL6pv375q0aKFLMuKZy4AAIBaibnMvP7668rLy9OAAQPimQcAACAqMc+Z8Xq9ateuXTyzAAAARC3mMjNr1izNnj1bp0+fjmceAACAqMR8munuu+/WypUrlZGRobZt26pRo0aVXi8qKqpzOAAAgHOJucyMGjVK27Zt04gRI5gADAAAHBNzmXnnnXe0du1a9e7dO555AAAAohLznBm/38/jAQAAgONiLjMLFizQlClTtG/fvjjGAYDa27Rpk+65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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "After removal of outlier using IQR Method"
      ],
      "metadata": {
        "id": "dzdutIsmegSR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "Q1=X1['Annual_income'].quantile(0.25)\n",
        "Q3=X1['Annual_income'].quantile(0.75)\n",
        "IQR = Q3 - Q1\n",
        "lower = Q1 - 1.5*IQR\n",
        "upper = Q3 + 1.5*IQR\n",
        "\n",
        "X1['Annual_income']=np.where(X1['Annual_income']<lower,lower,X1['Annual_income'])\n",
        "X1['Annual_income']=np.where(X1['Annual_income']>upper,upper,X1['Annual_income'])"
      ],
      "metadata": {
        "id": "wSPXGZeqDRXg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sns.boxplot(X1['Annual_income'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 447
        },
        "id": "hD6JbwFJGqo3",
        "outputId": "ef1727c3-8d18-496d-c0dc-584b866b846e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<Axes: >"
            ]
          },
          "metadata": {},
          "execution_count": 74
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Beforing using SMOTE"
      ],
      "metadata": {
        "id": "rvXdUBCV0q71"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y.value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FqZMkzCmgCs0",
        "outputId": "a0954daf-1ce5-4f25-a592-1b2307040536"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    1373\n",
              "1     175\n",
              "Name: label, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Treating Imbalance data  using SMOTE\n"
      ],
      "metadata": {
        "id": "G0VSdbYigbP-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from imblearn.over_sampling import SMOTE\n",
        "X,y=SMOTE().fit_resample(X,y)"
      ],
      "metadata": {
        "id": "XFmTj3uogFu6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## After Using SMOTE"
      ],
      "metadata": {
        "id": "Qis9-3fd0vqj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y.value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "byhYVkzjglCa",
        "outputId": "84af6e5b-537f-404d-a6bf-670b2e13be8f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1    1373\n",
              "0    1373\n",
              "Name: label, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 77
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Training the Model"
      ],
      "metadata": {
        "id": "evwgRQxogVoZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 70% training and 30% test"
      ],
      "metadata": {
        "id": "L9DelTyDVeDT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Feature scaling"
      ],
      "metadata": {
        "id": "ZJepzTI3g0zI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "\n",
        "sc=StandardScaler()\n",
        "\n",
        "X_train=sc.fit_transform(X_train)\n",
        "X_test=sc.fit_transform(X_test)\n"
      ],
      "metadata": {
        "id": "a56gl-wgRP8_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report\n",
        "\n",
        "\n",
        "lr=LogisticRegression()\n",
        "\n",
        "lr.fit(X_train,y_train)\n",
        "y_pred_train1=lr.predict(X_train)\n",
        "\n",
        "\n",
        "y_pred1=lr.predict(X_test)\n",
        "acc1=accuracy_score(y_test,y_pred1)\n",
        "\n",
        "pre1=precision_score(y_test,y_pred1)\n",
        "rec1=recall_score(y_test,y_pred1)\n",
        "f1sc1=f1_score(y_test,y_pred1)\n",
        "\n",
        "print(\"Training Accuracy of Decision Tree Classifier : \",accuracy_score(y_pred_train1,y_train))\n",
        "\n",
        "print(\"Accuracy of XGBClassifier : \",acc1)\n",
        "print(\"Precision : \",pre1)\n",
        "print(\"Recall : \",rec1)\n",
        "print(\"f1 score : \",f1sc1)\n",
        "print(confusion_matrix(y_test,y_pred1))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lVaUlnh_RlxA",
        "outputId": "da462a23-1d22-4664-a7ff-a4a7b4ff6741"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Accuracy of Decision Tree Classifier :  0.7108378870673953\n",
            "Accuracy of XGBClassifier :  0.7290909090909091\n",
            "Precision :  0.7358490566037735\n",
            "Recall :  0.7116788321167883\n",
            "f1 score :  0.7235621521335807\n",
            "[[206  70]\n",
            " [ 79 195]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "\n",
        "dt=DecisionTreeClassifier()\n",
        "\n",
        "\n",
        "dt.fit(X_train,y_train)\n",
        "\n",
        "y_pred_train2=dt.predict(X_train)\n",
        "y_pred2=dt.predict(X_test)\n",
        "\n",
        "print(\"Training Accuracy of Decision Tree Classifier : \",accuracy_score(y_pred_train2,y_train))\n",
        "\n",
        "acc2=accuracy_score(y_test,y_pred2)\n",
        "\n",
        "pre2=precision_score(y_test,y_pred2)\n",
        "rec2=recall_score(y_test,y_pred2)\n",
        "f1sc2=f1_score(y_test,y_pred2)\n",
        "print(\"Accuracy of XGBClassifier : \",acc2)\n",
        "print(\"Precision : \",pre2)\n",
        "print(\"Recall : \",rec2)\n",
        "print(\"f1 score : \",f1sc2)\n",
        "print(confusion_matrix(y_test,y_pred2))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7_j8e7wrSdLI",
        "outputId": "49b2fd31-c7c5-47d7-d6ff-b7041cc05fa1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Accuracy of Decision Tree Classifier :  0.9995446265938069\n",
            "Accuracy of XGBClassifier :  0.68\n",
            "Precision :  0.6484848484848484\n",
            "Recall :  0.781021897810219\n",
            "f1 score :  0.7086092715231788\n",
            "[[160 116]\n",
            " [ 60 214]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "rf=RandomForestClassifier()\n",
        "\n",
        "rf.fit(X_train,y_train)\n",
        "y_pred3=rf.predict(X_test)\n",
        "\n",
        "y_pred_train3=rf.predict(X_train)\n",
        "\n",
        "print(\"Training Accuracy of Random Forest Classifier : \",accuracy_score(y_pred_train3,y_train))\n",
        "\n",
        "acc3=accuracy_score(y_test,y_pred3)\n",
        "pre3=precision_score(y_test,y_pred3)\n",
        "rec3=recall_score(y_test,y_pred3)\n",
        "f1sc3=f1_score(y_test,y_pred3)\n",
        "print(\"Accuracy of XGBClassifier : \",acc3)\n",
        "print(\"Precision : \",pre3)\n",
        "print(\"Recall : \",rec3)\n",
        "print(\"f1 score : \",f1sc3)\n",
        "print(confusion_matrix(y_test,y_pred3))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LSz_Wgv8S3o-",
        "outputId": "8c975cc7-af5f-4cdf-e41d-712caf3f5d6f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Accuracy of Random Forest Classifier :  0.9995446265938069\n",
            "Accuracy of XGBClassifier :  0.8781818181818182\n",
            "Precision :  0.839344262295082\n",
            "Recall :  0.9343065693430657\n",
            "f1 score :  0.8842832469775475\n",
            "[[227  49]\n",
            " [ 18 256]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from xgboost import XGBClassifier\n",
        "xg=XGBClassifier()\n",
        "\n",
        "xg.fit(X_train,y_train)\n",
        "y_pred4=xg.predict(X_test)\n",
        "\n",
        "y_pred_train4=xg.predict(X_train)\n",
        "\n",
        "\n",
        "print(\"Training Accuracy of XGBClassifier : \",accuracy_score(y_pred_train4,y_train))\n",
        "\n",
        "acc4=accuracy_score(y_test,y_pred4)\n",
        "pre4=precision_score(y_test,y_pred4)\n",
        "rec4=recall_score(y_test,y_pred4)\n",
        "f1sc4=f1_score(y_test,y_pred4)\n",
        "print(\"Accuracy of XGBClassifier : \",acc4)\n",
        "print(\"Precision : \",pre4)\n",
        "print(\"Recall : \",rec4)\n",
        "print(\"f1 score : \",f1sc4)\n",
        "print(confusion_matrix(y_test,y_pred4))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qbRwvg3sTVf5",
        "outputId": "7bb6550a-7550-400c-810d-f6799a8cba51"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Accuracy of XGBClassifier :  0.9972677595628415\n",
            "Accuracy of XGBClassifier :  0.6381818181818182\n",
            "Precision :  0.5850340136054422\n",
            "Recall :  0.9416058394160584\n",
            "f1 score :  0.7216783216783217\n",
            "[[ 93 183]\n",
            " [ 16 258]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Data=pd.DataFrame({\"Models\":['LOGISTIC REGRESSION','DECISION TREE CLASSIFIER','RANDOM FOREST CLASSIFIER','XGBOOST CLASSIFIER'],\n",
        "                   \"Accuracy\":[acc1,acc2,acc3,acc4],\n",
        "                  \"Precision\":[pre1,pre2,pre3,pre4],\n",
        "                   \"Recall\":[rec1,rec2,rec3,rec4],\n",
        "                   \"f1-score\":[f1sc1,f1sc2,f1sc3,f1sc4]},index=[1,2,3,4])\n",
        "\n",
        "Data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 175
        },
        "id": "kkreUHvtRjvA",
        "outputId": "326a957a-1a50-47b3-ebed-fc9d5fef68f9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                     Models  Accuracy  Precision    Recall  f1-score\n",
              "1       LOGISTIC REGRESSION  0.729091   0.735849  0.711679  0.723562\n",
              "2  DECISION TREE CLASSIFIER  0.680000   0.648485  0.781022  0.708609\n",
              "3  RANDOM FOREST CLASSIFIER  0.878182   0.839344  0.934307  0.884283\n",
              "4        XGBOOST CLASSIFIER  0.638182   0.585034  0.941606  0.721678"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-666da121-a646-4851-8e84-b808833b4789\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Models</th>\n",
              "      <th>Accuracy</th>\n",
              "      <th>Precision</th>\n",
              "      <th>Recall</th>\n",
              "      <th>f1-score</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>LOGISTIC REGRESSION</td>\n",
              "      <td>0.729091</td>\n",
              "      <td>0.735849</td>\n",
              "      <td>0.711679</td>\n",
              "      <td>0.723562</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>DECISION TREE CLASSIFIER</td>\n",
              "      <td>0.680000</td>\n",
              "      <td>0.648485</td>\n",
              "      <td>0.781022</td>\n",
              "      <td>0.708609</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>RANDOM FOREST CLASSIFIER</td>\n",
              "      <td>0.878182</td>\n",
              "      <td>0.839344</td>\n",
              "      <td>0.934307</td>\n",
              "      <td>0.884283</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>XGBOOST CLASSIFIER</td>\n",
              "      <td>0.638182</td>\n",
              "      <td>0.585034</td>\n",
              "      <td>0.941606</td>\n",
              "      <td>0.721678</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-666da121-a646-4851-8e84-b808833b4789')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-666da121-a646-4851-8e84-b808833b4789 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-666da121-a646-4851-8e84-b808833b4789');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-d1f34555-9468-4e58-bc7f-f47eef845e20\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-d1f34555-9468-4e58-bc7f-f47eef845e20')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-d1f34555-9468-4e58-bc7f-f47eef845e20 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "  <div id=\"id_af77addd-8028-460c-b06d-3cfc49c103de\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('Data')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_af77addd-8028-460c-b06d-3cfc49c103de button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('Data');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 211
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.figure(figsize=(12,8))\n",
        "sns.barplot(x=Data['Models'],y=Data['Accuracy'])\n",
        "plt.title(\"Accuracies before Hyper parameter tuning\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 735
        },
        "id": "sSPUgUBkSH_Y",
        "outputId": "7e12c438-ba49-4d07-dc4e-6cb2a974614a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Accuracies before Hyper parameter tuning')"
            ]
          },
          "metadata": {},
          "execution_count": 85
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1200x800 with 1 Axes>"
            ],
            "image/png": "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iy++SPM8MmIaaREZGamIiAiNHj1af/31l8vw5P1f+uc2Bsk1GN/KdZ7SPDPi2JKRkp/tcP38k/dPALgTcCYdAG6jBQsW6Pz582rcuLHb4VWrVnWcyWzVqpX69eunOXPmqEWLFurUqZMiIyN16tQpLViwQJMmTVL58uXVrl07ffLJJ+rdu7fWrl2rmjVrKiEhQUuWLFG3bt302GOPKXv27GrRooXeeecdeXh4KCIiQl9//XWa7mtOVqpUKUVERKhv3746ePCggoKCNHfuXLf3kk6YMEH333+/KlasqGeffVZFihTR/v379c033yg+Pt7t9IOCgvT+++/rqaeeUsWKFdW6dWvlyZNHBw4c0DfffKMaNWro3Xff1c6dO/XII4+oZcuWKl26tLJkyaL58+fr6NGjat269Q2Xw8/PTwsXLlT79u0VFRWl7777Tt98840GDhzouIy9Vq1a6tKli2JiYhQfH69HH31U3t7e2rVrl2bPnq23335bzZs3T3Ee5cuXV/v27fXhhx/qzJkzqlWrltauXaupU6eqSZMmjgcGpqZfv35asGCBGjZsqA4dOigyMlIJCQnasmWL5syZo/379ztev5WR2rVrpwkTJmj58uVOr7W63r59+9S4cWPVrVtXq1ev1meffaYnnnhC5cuXl/TPDyevv/66BgwYoP3796tJkybKli2b9u3bp/nz5+vZZ59V3759b7qdadmODRo00NixY1W3bl098cQTOnbsmCZOnKhixYpp8+bNaZpPRkwjLTw9PfXxxx+rXr16KlOmjDp27KgCBQro4MGDWr58uYKCgvTVV19J+ifQS9KgQYPUunVreXt7q1GjRrd0nSfPs2fPnoqOjpaXl5dat26dIceWjBQZGalmzZpp/PjxOnnypOMVbMlXWfybh2kCwG1zux8nDwB3s0aNGhk/Pz+TkJCQYp0OHToYb29vc+LECWOMMSdPnjQ9evQwBQoUMD4+PqZgwYKmffv2juHG/POapkGDBpkiRYoYb29vExoaapo3b2727NnjqHP8+HHTrFkzkzVrVpMzZ07TpUsXs3XrVrevSbr+lWLJtm3bZmrXrm0CAwNNcHCw6dy5s9m0aZPLNIwxZuvWraZp06YmR44cxs/Pz5QsWdK8+uqrjuHXv4It2fLly010dLTJnj278fPzMxEREaZDhw5m3bp1xhhjTpw4Ybp3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LEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAAACxBSAcAAAAAwBKEdAAAAAAALEFIBwAAAADAEoR0AAAAAAAsQUgHAAAAAMAShHQAAAAA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NDRUtWvX1rx585TSswuvnZ+np6fy5cunVq1a6cCBA071HnzwQbdtee6555zqLV++XA0bNlSePHnk5+eniIgItWrVSitXrnTUWbFihdM08uTJo/r162vLli0u28rdPOvWreuos2nTJjVu3Fh58+aVn5+fwsPD1apVKx07dsxRZ/78+apataqyZ8+ubNmyqUyZMk7bIy4uTjly5HCad1r3g6FDh7pdD/Hx8Y5tdbe5drt5e3srJCREderU0ZQpU5SUlORUNzw83O02fuONN5zqzZ07Vw8++KCyZ8+uwMBAlStXTsOGDdOpU6ckuW7DxMREvfHGGypVqpT8/f2VK1cuRUVF6eOPP3Zq5/XHgj/++EOdOnVS/vz55ePjo8KFC6tXr146efKkU73k/jBz5kyn8vHjxys8PNzteklP307+v5eXl8LCwvTss886ljWt627//v0pzufnn392v/H+v+XLl6t+/frKnTu3smbNqtKlS6tPnz46ePCgpP/rw2fOnEl1OpIUHR0tLy8v/fLLLy7Djh8/rq5du6pQoULy9fVVaGiooqOj9dNPPznq3KiPJy9nfHx8qsvdtm3bdNW/dj2l9Thpi+v7YJEiRdS/f3/9/fffLnX//PNP+fj4pPj2Fg8PD/n5+en33393Km/SpIk6dOiQ4jxT6/eStGrVKtWvX185c+aUn5+fypYtq7FjxyoxMdFl/u722UuXLil37tzy8PDQihUr0rQurv3s3r3bUSe9/T55nZQoUUIxMTFOf1/Tsj/d6Ph0o+PE0KFDU1ze3bt3q2PHjipYsKB8fX1VpEgRtWnTRuvWrXNap1988UWK00gWExMjLy8vvfXWWy7D0nKMTUv/Dg8P1/jx453+f/3yFixYMF31UzsO5sqVS7Vq1dIPP/xww+W/0yUmJqp69ep6/PHHncrPnj2rsLAwDRo0yKl87ty5evjhh5UzZ075+/urZMmS6tSpkzZu3Oioc/3fscDAQEVGRmrevHku8//6669Vq1YtZcuWTVmzZlXlypUVFxfntq1Tp05V5cqVlTVrVmXLlk21atXS119/7VLvo48+Uvny5RUYGKgcOXLovvvuc7w+OqV9Iflz7fHqekeOHNHzzz+vokWLytfXV2FhYWrUqJGWLl3qqHP9vpeSGTNmyMvLS927d3c7PLVlkKQLFy5owIABioiIkJ+fn/LkyaNa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2bNVpkwZLV68+Kbm+2/PMGXLlk3x8fFat26dxowZo4oVK2rEiBEu9Z588knFx8c7fRo3buwYfv2vrdHR0YqPj9c333yjhIQElysHfvjhB61fv15xcXEqUaKE2/v6HnroIZd5XvtgixEjRujIkSOaNGmSypQpo0mTJqlUqVKOHwQCAgL0zTffaPfu3XrllVccB64qVarowoULKa6Tm1mnr7/+un744Qf973//S/e4dwtjjMt+0q9fP5dtXKlSJUf9m1G6dGlt3bpVP//8szp16qRjx46pUaNGeuaZZ27YvvTw9fXVsGHDNHr0aJ04ceKm2upOyZIlFR8fr19++UUvvfSSoqOjXe6xlFJfd8lmzZrlUicl7rbPzZoyZYpatWrl+OLXpk0b/fTTT9qzZ4+jTrNmzXTo0CEtWLBAdevW1YoVK1SxYkWnhwvdqI+n5PrlLl26dLrqX7+e0nqctEXysXPNmjVq3769OnbsqGbNmjmGnzlzRvPmzXN8oZRcf0i5VunSpdWuXbubOpvubr9Kb19r27atVq9erb179youLk6dOnVK87jX/x25PvSlpy3Jfwd/+ukn1atXT4MGDVL16tVd6qW2P93s8Sk1GXmlyYwZMxQREeE421qhQgUVLlxYs2bNctRJyzKkpX+7c/1xrV27dumqn9JxcOPGjZo7d66KFSumuLi4dP/4eyebMmWKsmbNqn379unPP/+8Yf1OnTopPj5eH3zwgRISEpz2r+RjYXx8vDZu3KiRI0fqueee01dffXVTbUvrvpsvXz6tXr1aW7ZsUa9evXT16lW1b99edevWvemgnpH9ZvHixUpISFD9+vUlScHBwY6HZyZLyzI88MAD2rt3r5YuXarmzZvr119/Vc2aNTV8+PBU53/9d/TrnyVwo/rXf6eXpHHjxik+Pl7fffedSpcurY8//li5cuW6mdVjLUL6HaJEiRLavn2722HJ5SVKlFCePHmUI0cOp4dkSFKhQoVUrFgxZcuW7YbzypYtmxo1aqQRI0Zo06ZNqlmzpl5//fWbandye3777bebGt/T01PFihXTPffco969e6tq1apuH1KVPXt2FStWzOmTvKzFixfX2bNndeTIEUf9wMBAFStWTIULF3Y73yJFiqhkyZJq3769nnnmGbVq1cqlTkBAgMs8rz9A5M6dWy1atNDo0aO1fft25c+fX6NH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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# K Fold Cross-Validation"
      ],
      "metadata": {
        "id": "hLoLA28lQ_H8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report\n",
        "from sklearn.model_selection import cross_val_score,StratifiedKFold\n",
        "from sklearn.svm import SVC\n"
      ],
      "metadata": {
        "id": "5jRj1NRVTJGU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=10)\n",
        "\n",
        "print(\"Logistic Regression Classifier  Accuracy is\",round((cross_val_score(lr,X,y,cv=skf,scoring=\"accuracy\").mean())*100,2))\n",
        "print(classification_report(y_test,y_pred1))"
      ],
      "metadata": {
        "id": "0nGO2WR2U8Hk",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "086e923e-ce38-46f5-b2c0-153e673496c1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Logistic Regression Classifier  Accuracy is 50.33\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.72      0.75      0.73       276\n",
            "           1       0.74      0.71      0.72       274\n",
            "\n",
            "    accuracy                           0.73       550\n",
            "   macro avg       0.73      0.73      0.73       550\n",
            "weighted avg       0.73      0.73      0.73       550\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Decision Tree Classifier  Accuracy is\",round((cross_val_score(dt,X,y,cv=skf,scoring=\"accuracy\").mean())*100,2))\n",
        "print(classification_report(y_test,y_pred2))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "txHhI93lUHMI",
        "outputId": "d4db6cc2-d5ef-4a85-b11f-0ddd9b17a753"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Decision Tree Classifier  Accuracy is 90.46\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.75      0.61      0.67       276\n",
            "           1       0.67      0.80      0.73       274\n",
            "\n",
            "    accuracy                           0.70       550\n",
            "   macro avg       0.71      0.70      0.70       550\n",
            "weighted avg       0.71      0.70      0.70       550\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Random Forest Classifier Accuracy is\",round((cross_val_score(rf,X,y,cv=skf,scoring=\"accuracy\").mean())*100,2))\n",
        "print(classification_report(y_test,y_pred3))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xlg_D36yVZ-P",
        "outputId": "3fc99c58-e6c4-42a3-9cb9-99cd2a793715"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Random Forest Classifier Accuracy is 95.99\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.92      0.80      0.85       276\n",
            "           1       0.82      0.93      0.87       274\n",
            "\n",
            "    accuracy                           0.86       550\n",
            "   macro avg       0.87      0.86      0.86       550\n",
            "weighted avg       0.87      0.86      0.86       550\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"XGBoost Classifier Accuracy is\",round((cross_val_score(xg,X,y,cv=skf,scoring=\"accuracy\").mean())*100,2))\n",
        "print(classification_report(y_test,y_pred4))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oggTcB-uU_ea",
        "outputId": "7e6adc19-cf66-491b-e337-9134b44d4164"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "XGBoost Classifier Accuracy is 95.56\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.85      0.34      0.48       276\n",
            "           1       0.59      0.94      0.72       274\n",
            "\n",
            "    accuracy                           0.64       550\n",
            "   macro avg       0.72      0.64      0.60       550\n",
            "weighted avg       0.72      0.64      0.60       550\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Hyper Parameter Tuning"
      ],
      "metadata": {
        "id": "z4_ddHZgnuwz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# ___Grid Search CV____"
      ],
      "metadata": {
        "id": "n3_w7kKOV5Mb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Logistic Regression"
      ],
      "metadata": {
        "id": "Ljx7f0pgnSAf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "param_grid=[{'penalty':['l1','l2','elasticnet','none'],\n",
        "            'C' : np.logspace(-4,4,20),\n",
        "            'solver': ['lbfgs','newton-cg','liblinear','sag','sage'],\n",
        "            'max_iter':[100,1000,2500,5000]\n",
        "          }]\n",
        "print(param_grid)\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "log_grid=GridSearchCV(estimator=lr,param_grid=param_grid,cv=3,verbose=True,n_jobs=-1)\n",
        "log_grid.fit(X_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 242
        },
        "id": "WZ28dj6VW44w",
        "outputId": "a12eac97-8039-489b-b12d-fc8a56e65285"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[{'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'C': array([1.00000000e-04, 2.63665090e-04, 6.95192796e-04, 1.83298071e-03,\n",
            "       4.83293024e-03, 1.27427499e-02, 3.35981829e-02, 8.85866790e-02,\n",
            "       2.33572147e-01, 6.15848211e-01, 1.62377674e+00, 4.28133240e+00,\n",
            "       1.12883789e+01, 2.97635144e+01, 7.84759970e+01, 2.06913808e+02,\n",
            "       5.45559478e+02, 1.43844989e+03, 3.79269019e+03, 1.00000000e+04]), 'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'sage'], 'max_iter': [100, 1000, 2500, 5000]}]\n",
            "Fitting 3 folds for each of 1600 candidates, totalling 4800 fits\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GridSearchCV(cv=3, estimator=LogisticRegression(), n_jobs=-1,\n",
              "             param_grid=[{'C': array([1.00000000e-04, 2.63665090e-04, 6.95192796e-04, 1.83298071e-03,\n",
              "       4.83293024e-03, 1.27427499e-02, 3.35981829e-02, 8.85866790e-02,\n",
              "       2.33572147e-01, 6.15848211e-01, 1.62377674e+00, 4.28133240e+00,\n",
              "       1.12883789e+01, 2.97635144e+01, 7.84759970e+01, 2.06913808e+02,\n",
              "       5.45559478e+02, 1.43844989e+03, 3.79269019e+03, 1.00000000e+04]),\n",
              "                          'max_iter': [100, 1000, 2500, 5000],\n",
              "                          'penalty': ['l1', 'l2', 'elasticnet', 'none'],\n",
              "                          'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag',\n",
              "                                     'sage']}],\n",
              "             verbose=True)"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=3, estimator=LogisticRegression(), n_jobs=-1,\n",
              "             param_grid=[{&#x27;C&#x27;: array([1.00000000e-04, 2.63665090e-04, 6.95192796e-04, 1.83298071e-03,\n",
              "       4.83293024e-03, 1.27427499e-02, 3.35981829e-02, 8.85866790e-02,\n",
              "       2.33572147e-01, 6.15848211e-01, 1.62377674e+00, 4.28133240e+00,\n",
              "       1.12883789e+01, 2.97635144e+01, 7.84759970e+01, 2.06913808e+02,\n",
              "       5.45559478e+02, 1.43844989e+03, 3.79269019e+03, 1.00000000e+04]),\n",
              "                          &#x27;max_iter&#x27;: [100, 1000, 2500, 5000],\n",
              "                          &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, &#x27;elasticnet&#x27;, &#x27;none&#x27;],\n",
              "                          &#x27;solver&#x27;: [&#x27;lbfgs&#x27;, &#x27;newton-cg&#x27;, &#x27;liblinear&#x27;, &#x27;sag&#x27;,\n",
              "                                     &#x27;sage&#x27;]}],\n",
              "             verbose=True)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=3, estimator=LogisticRegression(), n_jobs=-1,\n",
              "             param_grid=[{&#x27;C&#x27;: array([1.00000000e-04, 2.63665090e-04, 6.95192796e-04, 1.83298071e-03,\n",
              "       4.83293024e-03, 1.27427499e-02, 3.35981829e-02, 8.85866790e-02,\n",
              "       2.33572147e-01, 6.15848211e-01, 1.62377674e+00, 4.28133240e+00,\n",
              "       1.12883789e+01, 2.97635144e+01, 7.84759970e+01, 2.06913808e+02,\n",
              "       5.45559478e+02, 1.43844989e+03, 3.79269019e+03, 1.00000000e+04]),\n",
              "                          &#x27;max_iter&#x27;: [100, 1000, 2500, 5000],\n",
              "                          &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;, &#x27;elasticnet&#x27;, &#x27;none&#x27;],\n",
              "                          &#x27;solver&#x27;: [&#x27;lbfgs&#x27;, &#x27;newton-cg&#x27;, &#x27;liblinear&#x27;, &#x27;sag&#x27;,\n",
              "                                     &#x27;sage&#x27;]}],\n",
              "             verbose=True)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 93
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(log_grid.best_estimator_)\n",
        "print(f\"Accuracy of Logistic Regresssion after Hyper Parameter Tuning  is {log_grid.best_score_:.2f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dd4Pm8kzXnjb",
        "outputId": "ea819664-9ee8-486f-f7cf-ff070a62ea16"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "LogisticRegression(C=0.004832930238571752, solver='liblinear')\n",
            "Accuracy of Logistic Regresssion after Hyper Parameter Tuning  is 0.72\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# DecisionTreeClassifier"
      ],
      "metadata": {
        "id": "eVs1XazVnZK7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "params =  {\n",
        "    'min_samples_leaf': [1, 2, 3],\n",
        "    'max_depth': [1, 2, 3]\n",
        "}\n",
        "# Create gridsearch instance\n",
        "dt_grid = GridSearchCV(estimator=dt,\n",
        "                    param_grid=params,\n",
        "                    cv=10,\n",
        "                    n_jobs=-1,\n",
        "                    verbose=2)\n",
        "# Fit the model\n",
        "dt_grid.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 135
        },
        "id": "pcjsxkAfXwLg",
        "outputId": "84db7ad4-43a6-45b0-b638-0c6e70c4eb43"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fitting 10 folds for each of 9 candidates, totalling 90 fits\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GridSearchCV(cv=10, estimator=DecisionTreeClassifier(), n_jobs=-1,\n",
              "             param_grid={'max_depth': [1, 2, 3], 'min_samples_leaf': [1, 2, 3]},\n",
              "             verbose=2)"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=10, estimator=DecisionTreeClassifier(), n_jobs=-1,\n",
              "             param_grid={&#x27;max_depth&#x27;: [1, 2, 3], &#x27;min_samples_leaf&#x27;: [1, 2, 3]},\n",
              "             verbose=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=10, estimator=DecisionTreeClassifier(), n_jobs=-1,\n",
              "             param_grid={&#x27;max_depth&#x27;: [1, 2, 3], &#x27;min_samples_leaf&#x27;: [1, 2, 3]},\n",
              "             verbose=2)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 95
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(dt_grid.best_estimator_)\n",
        "print(dt_grid.best_params_)\n",
        "print(f\"Accuracy of Decision Tree Classifier after Hyper Parameter Tuning is {dt_grid.best_score_:.2f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bosTwfiRYNI7",
        "outputId": "041746dc-d020-46ea-f8ef-a647d42ca588"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "DecisionTreeClassifier(max_depth=3)\n",
            "{'max_depth': 3, 'min_samples_leaf': 1}\n",
            "Accuracy of Decision Tree Classifier after Hyper Parameter Tuning is 0.73\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Random Forest Classifier"
      ],
      "metadata": {
        "id": "RZrCFdK7nfNs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Number of trees in random forest\n",
        "n_estimators=[20,60,100,120]\n",
        "\n",
        "# Number of features to consider at every split\n",
        "max_features=[0.2,0.6,1.0]\n",
        "\n",
        "# Maximum number of levels in tree\n",
        "max_depth=[2,8,None]\n",
        "\n",
        "# Number of samples\n",
        "max_samples=[0.5,0.75,1.0]\n",
        "param_grid={'n_estimators':n_estimators,\n",
        "            'max_features':max_features,\n",
        "            'max_depth':max_depth,\n",
        "\n",
        "            'max_samples': max_samples\n",
        "          }\n",
        "print(param_grid)\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "rf_grid=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,verbose=2,n_jobs=-1)\n",
        "rf_grid.fit(X_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 152
        },
        "id": "tL__HNFrTHgi",
        "outputId": "2bb4d29f-6bf7-46ac-c4ba-1f9f3556f258"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'n_estimators': [20, 60, 100, 120], 'max_features': [0.2, 0.6, 1.0], 'max_depth': [2, 8, None], 'max_samples': [0.5, 0.75, 1.0]}\n",
            "Fitting 5 folds for each of 108 candidates, totalling 540 fits\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=-1,\n",
              "             param_grid={'max_depth': [2, 8, None],\n",
              "                         'max_features': [0.2, 0.6, 1.0],\n",
              "                         'max_samples': [0.5, 0.75, 1.0],\n",
              "                         'n_estimators': [20, 60, 100, 120]},\n",
              "             verbose=2)"
            ],
            "text/html": [
              "<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=-1,\n",
              "             param_grid={&#x27;max_depth&#x27;: [2, 8, None],\n",
              "                         &#x27;max_features&#x27;: [0.2, 0.6, 1.0],\n",
              "                         &#x27;max_samples&#x27;: [0.5, 0.75, 1.0],\n",
              "                         &#x27;n_estimators&#x27;: [20, 60, 100, 120]},\n",
              "             verbose=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=-1,\n",
              "             param_grid={&#x27;max_depth&#x27;: [2, 8, None],\n",
              "                         &#x27;max_features&#x27;: [0.2, 0.6, 1.0],\n",
              "                         &#x27;max_samples&#x27;: [0.5, 0.75, 1.0],\n",
              "                         &#x27;n_estimators&#x27;: [20, 60, 100, 120]},\n",
              "             verbose=2)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier()</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier()</pre></div></div></div></div></div></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 97
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(rf_grid.best_params_)\n",
        "print(f\"Accuracy of Random Forest Classifier after Hyper Parameter Tuning  is {rf_grid.best_score_:.2f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Gvp2hNYCS9Vd",
        "outputId": "6241285c-36fe-4462-ca69-0a09352ede0c"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'max_depth': None, 'max_features': 0.6, 'max_samples': 1.0, 'n_estimators': 120}\n",
            "Accuracy of Random Forest Classifier after Hyper Parameter Tuning  is 0.95\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# XGBClassifier"
      ],
      "metadata": {
        "id": "bvvtVhnJnnvN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "param_test = {'max_depth':[3,5,6,7,9]}\n",
        "model = XGBClassifier(learning_rate=0.3,\n",
        "                     n_estimators=16,\n",
        "                     max_depth=6,\n",
        "                     min_child_weight=1,\n",
        "                     gamma=0,\n",
        "                     subsample=1,\n",
        "                     colsample_bytree=1,\n",
        "                     objective='binary:logistic',\n",
        "                     nthread=4,\n",
        "                     scale_pos_weight=1,\n",
        "                     random_state=27)\n",
        "xgb_grid = GridSearchCV(estimator=model, param_grid = param_test, scoring='roc_auc', cv=5)\n",
        "xgb_grid.fit(X_train, y_train)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 118
        },
        "id": "fzXntBQK2te4",
        "outputId": "c1a75e3c-85c2-4557-871b-a8dc370f4627"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GridSearchCV(cv=5,\n",
              "             estimator=XGBClassifier(base_score=None, booster=None,\n",
              "                                     callbacks=None, colsample_bylevel=None,\n",
              "                                     colsample_bynode=None, colsample_bytree=1,\n",
              "                                     device=None, early_stopping_rounds=None,\n",
              "                                     enable_categorical=False, eval_metric=None,\n",
              "                                     feature_types=None, gamma=0,\n",
              "                                     grow_policy=None, importance_type=None,\n",
              "                                     interaction_constraints=None,\n",
              "                                     learning_rate=0.3, max_bin=None,\n",
              "                                     max_cat_threshold=None,\n",
              "                                     max_cat_to_onehot=None,\n",
              "                                     max_delta_step=None, max_depth=6,\n",
              "                                     max_leaves=None, min_child_weight=1,\n",
              "                                     missing=nan, monotone_constraints=None,\n",
              "                                     multi_strategy=None, n_estimators=16,\n",
              "                                     n_jobs=None, nthread=4,\n",
              "                                     num_parallel_tree=None, ...),\n",
              "             param_grid={'max_depth': [3, 5, 6, 7, 9]}, scoring='roc_auc')"
            ],
            "text/html": [
              "<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=5,\n",
              "             estimator=XGBClassifier(base_score=None, booster=None,\n",
              "                                     callbacks=None, colsample_bylevel=None,\n",
              "                                     colsample_bynode=None, colsample_bytree=1,\n",
              "                                     device=None, early_stopping_rounds=None,\n",
              "                                     enable_categorical=False, eval_metric=None,\n",
              "                                     feature_types=None, gamma=0,\n",
              "                                     grow_policy=None, importance_type=None,\n",
              "                                     interaction_constraints=None,\n",
              "                                     learning_rate=0.3, max_bin=None,\n",
              "                                     max_cat_threshold=None,\n",
              "                                     max_cat_to_onehot=None,\n",
              "                                     max_delta_step=None, max_depth=6,\n",
              "                                     max_leaves=None, min_child_weight=1,\n",
              "                                     missing=nan, monotone_constraints=None,\n",
              "                                     multi_strategy=None, n_estimators=16,\n",
              "                                     n_jobs=None, nthread=4,\n",
              "                                     num_parallel_tree=None, ...),\n",
              "             param_grid={&#x27;max_depth&#x27;: [3, 5, 6, 7, 9]}, scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" ><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=5,\n",
              "             estimator=XGBClassifier(base_score=None, booster=None,\n",
              "                                     callbacks=None, colsample_bylevel=None,\n",
              "                                     colsample_bynode=None, colsample_bytree=1,\n",
              "                                     device=None, early_stopping_rounds=None,\n",
              "                                     enable_categorical=False, eval_metric=None,\n",
              "                                     feature_types=None, gamma=0,\n",
              "                                     grow_policy=None, importance_type=None,\n",
              "                                     interaction_constraints=None,\n",
              "                                     learning_rate=0.3, max_bin=None,\n",
              "                                     max_cat_threshold=None,\n",
              "                                     max_cat_to_onehot=None,\n",
              "                                     max_delta_step=None, max_depth=6,\n",
              "                                     max_leaves=None, min_child_weight=1,\n",
              "                                     missing=nan, monotone_constraints=None,\n",
              "                                     multi_strategy=None, n_estimators=16,\n",
              "                                     n_jobs=None, nthread=4,\n",
              "                                     num_parallel_tree=None, ...),\n",
              "             param_grid={&#x27;max_depth&#x27;: [3, 5, 6, 7, 9]}, scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" ><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: XGBClassifier</label><div class=\"sk-toggleable__content\"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
              "              colsample_bylevel=None, colsample_bynode=None, colsample_bytree=1,\n",
              "              device=None, early_stopping_rounds=None, enable_categorical=False,\n",
              "              eval_metric=None, feature_types=None, gamma=0, grow_policy=None,\n",
              "              importance_type=None, interaction_constraints=None,\n",
              "              learning_rate=0.3, max_bin=None, max_cat_threshold=None,\n",
              "              max_cat_to_onehot=None, max_delta_step=None, max_depth=6,\n",
              "              max_leaves=None, min_child_weight=1, missing=nan,\n",
              "              monotone_constraints=None, multi_strategy=None, n_estimators=16,\n",
              "              n_jobs=None, nthread=4, num_parallel_tree=None, ...)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-15\" type=\"checkbox\" ><label for=\"sk-estimator-id-15\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">XGBClassifier</label><div class=\"sk-toggleable__content\"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
              "              colsample_bylevel=None, colsample_bynode=None, colsample_bytree=1,\n",
              "              device=None, early_stopping_rounds=None, enable_categorical=False,\n",
              "              eval_metric=None, feature_types=None, gamma=0, grow_policy=None,\n",
              "              importance_type=None, interaction_constraints=None,\n",
              "              learning_rate=0.3, max_bin=None, max_cat_threshold=None,\n",
              "              max_cat_to_onehot=None, max_delta_step=None, max_depth=6,\n",
              "              max_leaves=None, min_child_weight=1, missing=nan,\n",
              "              monotone_constraints=None, multi_strategy=None, n_estimators=16,\n",
              "              n_jobs=None, nthread=4, num_parallel_tree=None, ...)</pre></div></div></div></div></div></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 100
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Print the best score and the corresponding hyperparameters\n",
        "print(f'The best score of XGBoost Classifier after hyper parameter tuning is {xgb_grid.best_score_:.2f}')\n",
        "print(f'The best hyperparameters are {xgb_grid.best_params_}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0KMk-Ejzg7IK",
        "outputId": "f6fe3a27-d812-4f1f-ddb8-7f65b83c05cd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The best score of XGBoost Classifier after hyper parameter tuning is 0.98\n",
            "The best hyperparameters are {'max_depth': 9}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Accuracy_Final=pd.DataFrame({'Models':['Logistic Regression','Decision Tree Classifier','Random Forest Classifier','XGB Classifier'],\n",
        "                             'Accuracy':[log_grid.best_score_,dt_grid.best_score_,rf_grid.best_score_,xgb_grid.best_score_]},\n",
        "                            index=[1,2,3,4])\n",
        "\n"
      ],
      "metadata": {
        "id": "eoagbr4ChCYD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Accuracy_Final\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 175
        },
        "id": "KzVStekcq-LV",
        "outputId": "345f320d-fd16-4d59-aafe-601e418d4482"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                     Models  Accuracy\n",
              "1       Logistic Regression  0.719490\n",
              "2  Decision Tree Classifier  0.733101\n",
              "3  Random Forest Classifier  0.953086\n",
              "4            XGB Classifier  0.977914"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-acd8e400-f1bf-4a13-8a66-860c4441e140\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Models</th>\n",
              "      <th>Accuracy</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Logistic Regression</td>\n",
              "      <td>0.719490</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Decision Tree Classifier</td>\n",
              "      <td>0.733101</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Random Forest Classifier</td>\n",
              "      <td>0.953086</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>XGB Classifier</td>\n",
              "      <td>0.977914</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-acd8e400-f1bf-4a13-8a66-860c4441e140')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-acd8e400-f1bf-4a13-8a66-860c4441e140 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-acd8e400-f1bf-4a13-8a66-860c4441e140');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-3e5c317b-2380-4286-9b4b-1bc43a7d270b\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-3e5c317b-2380-4286-9b4b-1bc43a7d270b')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-3e5c317b-2380-4286-9b4b-1bc43a7d270b button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "  <div id=\"id_c6f8b1c5-1d0b-4c92-af76-51b20f7a050e\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('Accuracy_Final')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_c6f8b1c5-1d0b-4c92-af76-51b20f7a050e button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('Accuracy_Final');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 103
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,8))\n",
        "sns.barplot(x=Accuracy_Final['Models'],y=Accuracy_Final['Accuracy'])\n",
        "plt.title(\"Accuracies after Hyper parameter tuning\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 735
        },
        "id": "pUX444Twtczg",
        "outputId": "d02fd10d-a357-478f-ae32-8caa6bdc94b5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Accuracies after Hyper parameter tuning')"
            ]
          },
          "metadata": {},
          "execution_count": 104
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x800 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA04AAAK9CAYAAAAT0TyCAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABgQklEQVR4nO3deZyN9f//8efMmH2zjJmxjBn7kp3skgyjECWkxZIk2YokLSSfaLNEyoePpYXI+vFJKUQKRRhkN9bssg9mmHn//vCb83XM8jaacWge99vt3G4z72t7Xde5rnPO81zX9T5uxhgjAAAAAEC63F1dAAAAAADc6QhOAAAAAGBBcAIAAAAAC4ITAAAAAFgQnAAAAADAguAEAAAAABYEJwAAAACwIDgBAAAAgAXBCQAAAAAsCE4AkI2mTp0qNzc37du3z9WlpGnt2rWqU6eO/P395ebmptjYWFeXBNxV9u3bJzc3N02dOtXVpQDIZgQnANnuk08+kZubm2rWrOnqUnCdK1euqE2bNjp16pRGjRqlL774QpGRkfrkk09u+4fAqKgoNW/ePM1hy5cvl5ubm2bPnn1ba0LW+Pbbb/XWW2/d1mVOnz5do0ePvq3LBPDP52aMMa4uAsA/W926dXX48GHt27dPu3btUokSJVxd0m2TlJSkK1euyNvbW25ubq4ux8n27dtVtmxZTZw4Uc8++6yjvXz58goJCdHy5ctvWy1RUVEqX768vvnmm1TDli9froYNG2rWrFl67LHHbltNyBo9e/bUuHHjdDs/bjRv3lx//PHHbTnTa4xRQkKCPD095eHhke3LA+A6nHECkK327t2rVatWaeTIkcqfP7+mTZvm6pLSFR8fn+Xz9PDwkI+Pzx0XmiTp+PHjkqTcuXNn+7KuXr2qxMTEbF+OKxhjdOnSJZfW8E/evmm5E7Z5Cjc3N/n4+BCagByA4AQgW02bNk158uRRs2bN9Nhjj6UbnM6cOaOXXnpJUVFR8vb2VuHChdWhQwedPHnSMc7ly5f11ltvqVSpUvLx8VGBAgX06KOPKi4uTtL/XdJ145mStO5B6NSpkwICAhQXF6eHHnpIgYGBevLJJyVJP//8s9q0aaMiRYrI29tbEREReumll9L8oLZ9+3a1bdtW+fPnl6+vr0qXLq3XX3/dMTy9e5y+++471a9fX/7+/goMDFSzZs20ZcsWp3GOHj2qzp07q3DhwvL29laBAgXUsmVL67fomzZtUqdOnVSsWDH5+PgoPDxczzzzjP766y+n9W/QoIEkqU2bNnJzc9P999+vqKgobdmyRT/99JPc3Nwc7dc/Ty+++KIiIiLk7e2tEiVK6L333lNycnKq7f3hhx9q9OjRKl68uLy9vbV169YM675Zy5Ytk5ubm+bNm5dq2PTp0+Xm5qbVq1c71jMgIEB79uxRTEyM/P39VbBgQb399tupzoAkJydr9OjRuueee+Tj46OwsDB169ZNp0+fdhov5bLC77//XtWrV5evr6/+/e9/p1vv/fffr/Lly2vdunWqU6eOfH19VbRoUY0fP95pvMTERA0aNEjVqlVTcHCw/P39Vb9+fS1btsxpvIy2763MY9y4cSpWrJj8/PzUpEkTHTx4UMYYDR06VIULF5avr69atmypU6dOpVo3237cqVMnjRs3TpIc+9P1XyJkxza///77tXDhQu3fv9+xvKioKEnpH49pvXakPG9bt25Vw4YN5efnp0KFCun9999Pc1um9fpy6NAhtWrVSgEBAcqfP79efvllJSUlOU3/119/6emnn1ZQUJBy586tjh07auPGjdw3BdyBcrm6AAD/bNOmTdOjjz4qLy8vtW/fXp9++qnWrl2re++91zHOhQsXVL9+fW3btk3PPPOMqlatqpMnT2rBggX6888/FRISoqSkJDVv3lxLly7V448/rj59+uj8+fNavHix/vjjDxUvXjzTtV29elUxMTGqV6+ePvzwQ/n5+UmSZs2apYsXL6p79+7Kly+f1qxZo7Fjx+rPP//UrFmzHNNv2rRJ9evXl6enp5577jlFRUUpLi5O//vf//TOO++ku9wvvvhCHTt2VExMjN577z1dvHhRn376qerVq6cNGzY4PuS1bt1aW7ZsUa9evRQVFaXjx49r8eLFOnDggGOctCxevFh79uxR586dFR4eri1btmjChAnasmWLfv31V7m5ualbt24qVKiQhg0bpt69e+vee+9VWFiY4uPj1atXLwUEBDgCYFhYmCTp4sWLatCggQ4dOqRu3bqpSJEiWrVqlQYOHKgjR46kuqdkypQpunz5sp577jl5e3srb968GT4fV65ccQrKKc6ePev0//3336+IiAhNmzZNjzzyiNOwadOmqXjx4qpdu7ajLSkpSU2bNlWtWrX0/vvv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pqVLlzrakpOTtXTpUtWuXTvNaWrXru00viQtXrw43fEBAAAA4O9y+aV6ffv2VceOHVW9enXVqFFDo0ePVnx8vDp37ixJ6tChgwoVKqThw4dLkvr06aMGDRpoxIgRatasmWbMmKHff/9dEyZMcOVqAAAAAPgHc3lwateunU6cOKFBgwbp6NGjqly5shYtWuToAOLAgQNyd/+/E2N16tTR9OnT9cYbb+i1115TyZIlNX/+fJUvX95Vq4BM8vb21uDBg1NdQgnkJBwHAMcBwDFwd3EzN9P3HgAAAADkYC7/AVwAAAAAuNMRnAAAAADAguAEAAAAABYEp3+oqKgojR49+pannzp1Kr93lY6/u21x6zKz7XmenC1fvlxubm46c+ZMti/rrbfeUuXKlVO1hYWFyc3NTfPnz1enTp3UqlWrbK/lnyRl2yHr3a5tm9ZxOH/+fJUoUUIeHh568cUXef/NIe6//369+OKLt2VZN+7f27dvV61ateTj46PKlStr3759cnNzU2xs7G2p565mcNt17NjRtGzZMluXcfz4cRMfH39T40ZGRppRo0Y5tV28eNEcO3bslpc/ZcoUI8lIMm5ubiY8PNy0bdvW7N+//5bneafIzLbNCTp27Oh4rnPlymVCQ0NNdHS0mTRpkklKSsrSZWVm22f383T9eqf1iIyMzLZlp2X9+vXmscceM6Ghocbb29uUKFHCPPvss2bHjh3GGGOWLVtmJJnTp09ney3nz583J0+edPy/detWI8nMmzfPHDlyxFy+fNmcOXPmttSSlW7c16Oiokz//v3NpUuXbsvyU7bh7Zbevr5r167bXsv1Nd3s++iRI0dMz549TdGiRY2Xl5cpXLiwad68uVmyZIljnNu1bRMSEsyRI0dMcnKyoy00NNQMGDDAHDp0yJw7d+5vv//mRFevXjW1a9c2jzzyiFP7mTNnTOHChc1rr73m1D579mzTsGFDkzt3buPj42NKlSplOnfubNavX+8Y5/rPMZKMv7+/qVq1qpkzZ461noSEBPPee++ZihUrGl9fX5MvXz5Tp04dM3nyZJOYmGiMMaZBgwamT58+f3/lb0LK626Ktm3bmgceeMDs27fPnDx50ly9etUcOXLEXLly5bbUczfjjNM/VP78+eXn53fL0/v6+io0NPRv1RAUFKQjR47o0KFDmjNnjnbs2KE2bdr8rXnejCtXrmTr/P/utv0natq0qY4cOaJ9+/bpu+++U8OGDdWnTx81b95cV69ezbLlZGbbZ/fz9NFHH+nIkSOOhyRNmTLF8f/atWudxk9MTMy2Wr755hvVqlVLCQkJmjZtmrZt26Yvv/xSwcHBevPNN7NtuekJCAhQvnz5HP/HxcVJklq2bKnw8HB5e3srODj4b32rbozJ0n3rZqXs63v27NGoUaP073//W4MHD77tddxuKet9/aNo0aK3NK/sPBZutG/fPlWrVk0//vijPvjgA23evFmLFi1Sw4YN1aNHj9tWRwovLy+Fh4fLzc1NknThwgUdP35cMTExKliwoAIDA7Pk/Te73wfvNB4eHpo6daoWLVqkadOmOdp79eqlvHnzOh2jAwYMULt27VS5cmUtWLBAO3bs0PTp01WsWDENHDjQab4pn2OOHDmiDRs2KCYmRm3bttWOHTvSrSUxMVExMTF699139dxzz2nVqlVas2aNevToobFjx2rLli1ZvwEsUl53U8TFxalevXqKjIxUvnz55OHhofDwcOXKdeu/UnQ7j2uXcnVyy4ls35QtX77c3HvvvcbLy8uEh4ebAQMGOH0LcO7cOfPEE08YPz8/Ex4ebkaOHJnqm4vrzyIlJyebwYMHm4iICOPl5WUKFChgevXqZYy59o2HbvgW0Zhr37QEBwc71bVgwQJTvXp14+3tbfLly2datWqV7jqkNf2YMWOMJHP27FlH2/z5802VKlWMt7e3KVq0qHnrrbec1nXbtm2mbt26xtvb25QtW9YsXrzY6ZvBvXv3GklmxowZ5r777jPe3t5mypQpxhhjJk6caMqUKWO8vb1N6dKlzbhx4xzzTUhIMD169DDh4eHG29vbFClSxAwbNsy6vW7ctsYYs3//fvPwww8bf39/ExgYaNq0aWOOHj3qGD548GBTqVIl8/nnn5vIyEgTFBRk2rVrZ86dO5fu9rubpLc/L1261EgyEydOdLSdPn3adOnSxYSEhJjAwEDTsGFDExsb6zRdRvvZze7XN45rTPY/T7rhG+vIyEjz9ttvm6efftoEBgaajh07GmOM+fnnn029evWMj4+PKVy4sOnVq5e5cOGCY7rLly+bfv36mYIFCxo/Pz9To0YNs2zZsnSXGx8fb0JCQtI9HlPO6tx4xunkyZPm8ccfNwULFjS+vr6mfPnyZvr06U7Tzpo1y5QvX974+PiYvHnzmkaNGjlqXbZsmbn33nuNn5+fCQ4ONnXq1DH79u1z2pYpf6f1GnPjfpOUlGSGDRtmoqKijI+Pj6lYsaKZNWuWY3hK/d9++62pWrWq8fT0zHC7ZIe09vVHH33UVKlSxfH/zWzXBg0amF69epn+/fubPHnymLCwMDN48GCncXbu3Gnq16/veO374YcfUu1jmzZtMg0bNnQ8P127djXnz59PVe8777xjQkNDTXBwsBkyZIi5cuWKefnll02ePHlMoUKFzOTJkzO93tezvWc1aNDA9OjRw/Tp08fky5fP3H///cYYYzZv3myaNm1q/P39TWhoqHnqqafMiRMnHNOlt/+ltU+lty88+OCDplChQk7HWIrrz3jeuG1feeUVU7JkSePr62uKFi1q3njjDceZAmOMiY2NNffff78JCAgwgYGBpmrVqmbt2rXGGGP27dtnmjdvbnLnzm38/PxMuXLlzMKFC40xzsdhyt83rkda75+290pJ5pNPPjEtWrQwfn5+qfannOKjjz4yefLkMYcPHzbz5883np6eTu8xq1evNpLMRx99lOb0158JTOt5SEpKMp6enubrr79Ot4b33nvPuLu7O529SpGYmOjYF2/83Pb555+batWqmYCAABMWFmbat2/vdObx1KlT5oknnjAhISHGx8fHlChRwnHsZvSZxhjn/fvGfW7w4MGOz1IbNmxwTGM7PtM7rv/pCE4ukNGb0J9//mn8/PzMCy+8YLZt22bmzZtnQkJCnF4En332WRMZGWmWLFliNm/ebB555BETGBiYbnCaNWuWCQoKMt9++63Zv3+/+e2338yECROMMcb89ddfpnDhwubtt982R44cMUeOHDHGpH7B+Oabb4yHh4cZNGiQ2bp1q4mNjXU6KG904/THjh0zDRs2NB4eHo4XjRUrVpigoCAzdepUExcXZ3744QcTFRVl3nrrLWPMtVPvpUuXNo0bNzaxsbHm559/NjVq1EgzOEVFRZk5c+aYPXv2mMOHD5svv/zSFChQwNE2Z84ckzdvXjN16lRjjDEffPCBiYiIMCtWrDD79u0zP//8s+PDTUbb68Ztm5SUZCpXrmzq1atnfv/9d/Prr7+aatWqmQYNGjjGHzx4sAkICDCPPvqo2bx5s1mxYoUJDw9PdenA3Sqj/blSpUrmwQcfdPwfHR1tWrRoYdauXWt27txp+vXrZ/Lly2f++usvY4x9P7vZ/frGcW/H85RWcAoKCjIffvih2b17t+Ph7+9vRo0aZXbu3GlWrlxpqlSpYjp16uSY7tlnnzV16tQxK1asMLt37zYffPCB8fb2Njt37kxzuXPnzjWSzKpVqzKs78bg9Oeff5oPPvjAbNiwwcTFxZkxY8YYDw8P89tvvxljjDl8+LDJlSuXGTlypNm7d6/ZtGmTGTdunDl//ry5cuWKCQ4ONi+//LLZvXu32bp1q5k6darjUtzrg9P58+cdl7xc/xpz437zr3/9y5QpU8YsWrTIxMXFmSlTphhvb2+zfPlyp/orVqxofvjhB7N7927HfnO73Fjz5s2bTXh4uKlZs6ajzbZdjbn2gSMoKMi89dZbZufOneazzz4zbm5u5ocffjDGXNtfy5cvbxo1amRiY2PNTz/9ZKpUqeK0j124cMEUKFDAsb8uXbrUFC1a1BHQU+oNDAw0PXr0MNu3bzeTJk0ykkxMTIx55513zM6dO83QoUONp6enOXjw4E2v9/Vu5j2rQYMGJiAgwPTv399s377dbN++3Zw+fdrkz5/fDBw40Gzbts2sX7/eNG7c2DRs2NAYk/H+d/78edO2bVvTtGlTxz6VkJCQqra//vrLuLm5ZfheleLG43fo0KFm5cqVZu/evWbBggUmLCzMvPfee47h99xzj3nqqafMtm3bzM6dO83XX3/t+IDerFkz07hxY7Np0yYTFxdn/ve//5mffvrJGON8HCYkJJgdO3YYSWbOnDmO9bjx/dP2XplSf2hoqJk8ebKJi4v7R1wWfyuSk5PN/fffbxo1amRCQ0PN0KFDnYb37t3bBAQE3NQlaTc+D1evXjWTJ082np6eZvfu3elOV7FiRdOkSRPr/G8MTpMmTTLffvutiYuLM6tXrza1a9d2ev/s0aOHqVy5slm7dq3Zu3evWbx4sVmwYIExJuPPNMY4799Hjhwx99xzj+nXr585cuSIOX/+fKrgZDs+U+q/8bjOCQhOLpDRm9Brr71mSpcu7fStx7hx40xAQIBJSkoy586dM56enk7fxJ45c8b4+fmlG5xGjBhhSpUq5fRt2fXSusfpxheM2rVrmyeffPKm1zHlg5K/v7/x8/NzfLPRu3dvxziNGjVK9Yb2xRdfmAIFChhjjPnuu+9Mrly5HB+0jDHpnnEaPXq003yKFy+e6lveoUOHmtq1axtjjOnVq5d54IEHnLZzisxsrx9++MF4eHiYAwcOOIZv2bLFSDJr1qwxxlz7EOnn5+d05qJ///5OH7buZhntz+3atTNly5Y1xlw70xIUFOR0nbUx156rf//738YY+352q/v17Xie0gpON54F6tKli3nuueec2n7++Wfj7u5uLl26ZPbv3288PDzMoUOHnMZp1KiRGThwYJrLfe+994wkc+rUqQzru5l7nJo1a2b69etnjDFm3bp1RpLjLNL1/vrrLyPJEWpudH1wMsaYefPmOc40pbh+v7l8+bLx8/NLFf66dOli2rdv71T//PnzM1zP7NSxY0fj4eFh/P39jbe3t5Fk3N3dzezZszOc7vrtasy1Dxz16tVzGufee+81AwYMMMYY8/3335tcuXI57Qffffed0z42YcIEkydPHqczKQsXLjTu7u6OM6kdO3Y0kZGRTvcali5d2tSvX9/x/9WrV42/v7/56quvbmq9Ux6PPfaYMcb+npWyvteflTPm2uvxjR8uDx48aCSZHTt2ZLj/pdRku8fpt99+M5LM3LlzMxzPGPs9Th988IGpVq2a4//AwEDHF3E3qlChglOoud6Nx+Hp06dTnTG78f3X9l6ZUv+LL76Ybv05ybZt24wkU6FChVQBqWnTpqZixYpObSNGjHDat8+cOWOMcf4c4+/vb9zd3Z2uakmPr6+v02ed9NjucVq7dq2R5DiL3KJFC9O5c+c0x83oM40xqffvSpUqOX25cWNwsh2fKfXfeFznBLd+MSOyxbZt21S7dm3H9c+SVLduXV24cEF//vmnTp8+rStXrqhGjRqO4cHBwSpdunS682zTpo1Gjx6tYsWKqWnTpnrooYfUokWLTF3LGhsbq65du2ZqXQIDA7V+/XpduXJF3333naZNm6Z33nnHMXzjxo1auXKlU1tSUpIuX76sixcvaseOHYqIiFB4eLhj+PXrfb3q1as7/o6Pj1dcXJy6dOniVPPVq1cVHBwsSerUqZMaN26s0qVLq2nTpmrevLmaNGkiKXPba9u2bYqIiFBERISjrVy5csqdO7e2bdume++9V9K1Ht4CAwMd4xQoUEDHjx+/uQ15FzPGOPbljRs36sKFC073vkjSpUuXHPfAZGY/uxuep+v3S+naNti0aZPTNfjGGCUnJ2vv3r3as2ePkpKSVKpUKafpEhISUm2366e/FUlJSRo2bJi+/vprHTp0SImJiUpISHDcF1apUiU1atRIFSpUUExMjJo0aaLHHntMefLkUd68edWpUyfFxMSocePGio6OVtu2bVWgQIFbqmX37t26ePGiGjdu7NSemJioKlWqOLXduE1vt4YNG+rTTz9VfHy8Ro0apVy5cql169aO4bbtmqJixYpO/1+/r6XsrwULFnQMr127ttP427ZtU6VKleTv7+9oq1u3rpKTk7Vjxw6FhYVJku655x65u//f7cxhYWEqX768438PDw/ly5fPup+nrHeKlOXa3rOKFCkiSapWrZrT/DZu3Khly5YpICAg1bLi4uLUpEmTdPe/m3Wrx4YkzZw5U2PGjFFcXJwuXLigq1evKigoyDG8b9++evbZZ/XFF18oOjpabdq0UfHixSVJvXv3Vvfu3fXDDz8oOjparVu3TvV8Z4btvTJl33L1sXGnmDx5svz8/LR37179+eefioqKynD8Z555Rg8//LB+++03PfXUU077TcrnGEm6ePGilixZoueff1758uVTixYt0pzfre5369at01tvvaWNGzfq9OnTSk5OliQdOHBA5cqVU/fu3dW6dWutX79eTZo0UatWrVSnTh1JGX+muRW24zPlPerG4zonoHOIHCAiIkI7duzQJ598Il9fX73wwgu67777MnXzqK+vb6aX6+7urhIlSqhs2bLq27evatWqpe7duzuGX7hwQUOGDFFsbKzjsXnzZu3atUs+Pj6ZWtb1Hx4uXLggSZo4caLTvP/44w/9+uuvkqSqVatq7969Gjp0qC5duqS2bdvqsccek5Q12+tGnp6eTv+7ubk5XhT/ybZt2+a4gfzChQsqUKCA03MSGxurHTt2qH///pIyt5/dDc/T9fuldG0bdOvWzWn9N27cqF27dql48eK6cOGCPDw8tG7dOqdxtm3bpo8++ijNZaS8gW3fvj1TtX3wwQf66KOPNGDAAC1btkyxsbGKiYlx3ODr4eGhxYsX67vvvlO5cuU0duxYlS5dWnv37pV0rSOM1atXq06dOpo5c6ZKlSrlOL4yK+WYXbhwodN6b926VbNnz3Ya98Zterv5+/urRIkSqlSpkiZPnqzffvtNkyZNcgy3bdcUt+s1Ia3l3MqyU9Y75ZHZkJzWsdCiRYtUrwe7du3SfffdZ93/bkbJkiXl5uaW6WNj9erVevLJJ/XQQw/pm2++0YYNG/T66687PYdvvfWWtmzZombNmunHH39UuXLlNG/ePEnSs88+qz179ujpp5/W5s2bVb16dY0dOzZTNVzvZt8rXX1s3AlWrVqlUaNG6ZtvvlGNGjXUpUsXpyBTsmRJ7dmzx+l9Infu3CpRooQKFSqUan4pn2NKlCihihUrqm/fvrr//vv13nvvpVtDqVKlMr3PxcfHKyYmRkFBQZo2bZrWrl3r2J9S9rsHH3xQ+/fv10svvaTDhw+rUaNGevnllyVl/JnmVtiOzxQ5cZ8jON1hypYtq9WrVzsd6CtXrlRgYKAKFy6sYsWKydPT06nHrrNnz2rnzp0ZztfX11ctWrTQmDFjtHz5cq1evVqbN2+WdK2Xn6SkpAynr1ixopYuXfo31kx69dVXNXPmTMe3N1WrVtWOHTuc3oxTHu7u7ipdurQOHjyoY8eOOeZxY09laQkLC1PBggW1Z8+eVPO9vheooKAgtWvXThMnTtTMmTM1Z84cnTp1SlLG2+t6ZcuW1cGDB3Xw4EFH29atW3XmzBmVK1fulrfVP8GPP/6ozZs3O76Nr1q1qo4ePapcuXKlel5CQkIkZX4/u9uep6pVq2rr1q1p7vNeXl6qUqWKkpKSdPz48VTDrz/zer0mTZooJCRE77//fprD0/vdppUrV6ply5Z66qmnVKlSJRUrVizV64ibm5vq1q2rIUOGaMOGDfLy8nK8mUtSlSpVNHDgQK1atUrly5fX9OnTb2m7lCtXTt7e3jpw4ECq9b7+LOGdxt3dXa+99preeOMNXbp0SdLNbVeblP01pbdGSalCadmyZbVx40bFx8c72lauXOl47bxdbO9Z6alataq2bNmiqKioVM95yoexjPa/m3nfyps3r2JiYjRu3Din7ZQivWNj1apVioyM1Ouvv67q1aurZMmS2r9/f6rxSpUqpZdeekk//PCDHn30UU2ZMsUxLCIiQs8//7zmzp2rfv36aeLEiRnWmhHbeyWuuXjxojp16qTu3burYcOGmjRpktasWaPx48c7xmnfvr0uXLigTz755JaX4+Hh4Tje0/LEE09oyZIl2rBhQ6phV65cSXNf3L59u/766y+9++67ql+/vsqUKZPmWeD8+fOrY8eO+vLLLzV69GhNmDDBMSyjzzSZdTPHZ07FEeciZ8+eTZXkDx48qBdeeEEHDx5Ur169tH37dv33v//V4MGD1bdvX7m7uyswMFAdO3ZU//79tWzZMm3ZskVdunSRu7u706US15s6daomTZqkP/74Q3v27NGXX34pX19fRUZGSrp2edKKFSt06NAhnTx5Ms15DB48WF999ZUGDx6sbdu2afPmzRl+45KWiIgIPfLIIxo0aJAkadCgQfr88881ZMgQbdmyRdu2bdOMGTP0xhtvSJIaN26s4sWLq2PHjtq0aZNWrlzpGJbeuqYYMmSIhg8frjFjxmjnzp3avHmzpkyZopEjR0qSRo4cqa+++krbt2/Xzp07NWvWLIWHhyt37tzW7XW96OhoVahQQU8++aTWr1+vNWvWqEOHDmrQoEGOumwiISFBR48e1aFDh7R+/XoNGzZMLVu2VPPmzdWhQwdJ17ZV7dq11apVK/3www/at2+fVq1apddff12///67pMztZ3fj8zRgwACtWrVKPXv2dHx799///lc9e/aUdO2D2JNPPqkOHTpo7ty52rt3r9asWaPhw4dr4cKFac7T399f//nPf7Rw4UI9/PDDWrJkifbt26fff/9dr7zyip5//vk0pytZsqQWL16sVatWadu2berWrZvTlxS//fabhg0bpt9//10HDhzQ3LlzdeLECZUtW1Z79+7VwIEDtXr1au3fv18//PCDdu3apbJly97SdgkMDNTLL7+sl156SZ999pni4uK0fv16jR07Vp999tktzfN2adOmjTw8PDRu3DhJ9u16M6Kjo1WqVCl17NhRGzdu1M8//6zXX3/daZwnn3xSPj4+6tixo/744w8tW7ZMvXr10tNPP+24TO92sL1npadHjx46deqU2rdvr7Vr1youLk7ff/+9OnfurKSkpAz3P+na+9amTZu0Y8cOnTx5Mt0zzePGjVNSUpJq1KihOXPmaNeuXdq2bZvGjBmT6vLHFCVLltSBAwc0Y8YMxcXFacyYMU5fGFy6dEk9e/bU8uXLtX//fq1cuVJr16511Pbiiy/q+++/1969e7V+/XotW7bslo8Nyf5eiWsGDhwoY4zeffddSdf2kQ8//FCvvPKK9u3bJ+naJa/9+vVTv3791LdvX/3yyy/av3+/fv31V02aNElubm5O+60xRkePHtXRo0e1d+9eTZgwQd9//71atmyZbh0vvvii6tatq0aNGmncuHHauHGj9uzZo6+//lq1atXS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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# After Hyper parameter Tuning\n",
        "## By Comparing all the four models:\n",
        "\n",
        "Logistic Regression\t:- 74%\n",
        "\n",
        "Decision Tree Classifier:- 75%\n",
        "\n",
        "Random Forest Classifier:- 96%\n",
        "\n",
        "XGB Classifier:- 98%\n",
        "\n",
        "\n",
        " **XGB Classifier is the most appropriate model for Credit card Approval Process as it has highest accuracy of 98%.**"
      ],
      "metadata": {
        "id": "D_FOSV103kN9"
      }
    }
  ]
}