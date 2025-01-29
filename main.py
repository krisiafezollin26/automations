
#!/usr/bin/env python3

import os
import logging
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import gspread
from google.oauth2 import service_account

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class MAUForecastProcessor:
    def __init__(self, credentials_path: str, sheet_id: str):
        """
        Initialize the MAU forecast processor with Google Sheets credentials and sheet ID.
        Args:credentials_path: Path to the Google Sheets credentials JSON file
        sheet_id: ID of the Google Sheet to process
        """
        self.credentials_path = credentials_path
        self.sheet_id = sheet_id
        self.sheet = None
        self.worksheet = None
        
    def connect_to_sheets(self) -> None:
        """Establish connection to Google Sheets using service account credentials."""
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, 
                scopes=scopes)
            
            gc = gspread.authorize(creds)
            self.sheet = gc.open_by_key(self.sheet_id)
            self.worksheet = self.sheet.worksheet("actual_budget_2025")
            logger.info("Successfully connected to Google Sheets")
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {str(e)}")
            raise

    def get_mau_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieve NET and CUM MAU data from the worksheet.
        Returns:Tuple containing (net_mau, cum_mau) DataFrames
        """
        try:
            net_mau = pd.DataFrame(self.worksheet.get('A11:O19'))
            cum_mau = pd.DataFrame(self.worksheet.get('A1:O9'))
            logger.info("Successfully retrieved MAU data")
            return net_mau, cum_mau
        except Exception as e:
            logger.error(f"Failed to retrieve MAU data: {str(e)}")
            raise

    def prepare_converter_sheet(self, cum_mau: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the converter sheet with transposed data.
        Args:cum_mau: DataFrame containing cumulative MAU data     
        Returns:Processed DataFrame ready for forecasting
        """
        try:
            # Create or clear converter worksheet
            try:
                converter_worksheet = self.sheet.add_worksheet(
                    title="Converter", 
                    rows=100, 
                    cols=20)
        
            except gspread.exceptions.APIError:
                converter_worksheet = self.sheet.worksheet("Converter")
                converter_worksheet.clear()

            cum_mau_values = cum_mau.values.tolist()
            converter_worksheet.update("A1", cum_mau_values)
            
            converter_data = converter_worksheet.get_all_records()
            df_converter = pd.DataFrame(converter_data)
            df_converter = df_converter.rename(columns={"Cum. Net MAU": "Country"})
            
            logger.info("Successfully prepared converter sheet")
            return df_converter
        except Exception as e:
            logger.error(f"Failed to prepare converter sheet: {str(e)}")
            raise

    def format_and_forecast_data(self, df_converter: pd.DataFrame) -> pd.DataFrame:
        """
        Format data and generate forecasts through December 2027.
        Args:df_converter: Processed DataFrame from converter sheet     
        Returns:DataFrame containing historical and forecasted values
        """
        try:
            historical_months = df_converter.columns[1:].tolist()
            forecast_months = []
            for year in [26, 27]:
                for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                    forecast_months.append(f'{month} {year}')
            all_months = historical_months + forecast_months
            
            # Initialize result DataFrame
            result_df = pd.DataFrame({'Country': df_converter['Country']})
            
            # Generate forecasts for each country
            for country in df_converter['Country']:
                historical_values = df_converter[
                    df_converter['Country'] == country].iloc[:, 1:].values.flatten()
                
                # fit model
                X = np.arange(len(historical_values)).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, historical_values)
                
                # generate forecasted extrapolation
                future_X = np.arange(
                    len(historical_values),
                    len(historical_values) + len(forecast_months)).reshape(-1, 1)
                forecasted_values = model.predict(future_X)
                
                all_values = np.concatenate([historical_values, forecasted_values])
                
                country_mask = result_df['Country'] == country
                for month, value in zip(all_months, all_values):
                    result_df.loc[country_mask, month] = value
            
            logger.info("Successfully generated forecasts")
            return result_df
        except Exception as e:
            logger.error(f"Failed to generate forecasts: {str(e)}")
            raise

    def get_language_mapping(self) -> pd.DataFrame:
        """
        Retrieve language mapping data from the worksheet.
        Returns:DataFrame containing language mapping information
        """
        try:
            lang_table = pd.DataFrame(self.worksheet.get('A22:F29'))
            lang_table.columns = lang_table.iloc[0]
            lang_table = lang_table[1:].reset_index(drop=True)
            logger.info("Successfully retrieved language mapping")
            return lang_table
        except Exception as e:
            logger.error(f"Failed to retrieve language mapping: {str(e)}")
            raise

    def update_final_sheet(self, final_df: pd.DataFrame) -> None:
        """
        Update the final sheet with processed results.
        Args:final_df: DataFrame containing final results to be uploaded
        """
        try:
            final_df = final_df.T
            if "Country" in final_df.columns:
                final_df = final_df.drop(columns=["Country"])
            
            final_df.insert(0, "Country", final_df.index)
            final_df.columns = final_df.iloc[0]
            final_df = final_df[1:].reset_index(drop=True)
            final_df_values = [final_df.columns.tolist()] + final_df.values.tolist()

            try:
                final_worksheet = self.sheet.add_worksheet(
                    title="Final",
                    rows=100,
                    cols=20
                )
            except gspread.exceptions.APIError:
                final_worksheet = self.sheet.worksheet("Final")
                final_worksheet.clear()

            final_worksheet.update("A1", final_df_values)
            logger.info("Successfully updated final sheet")
        except Exception as e:
            logger.error(f"Failed to update final sheet: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize processor
        processor = MAUForecastProcessor(
            credentials_path="wfm-automations.json",
            sheet_id="1cbW6p8rdMhq9XD5WL-QJNO1UBRQRTcKQBoh6g1QfEv8")
        
        # execute processing steps
        processor.connect_to_sheets()
        net_mau, cum_mau = processor.get_mau_data()
        df_converter = processor.prepare_converter_sheet(cum_mau)
        final_df = processor.format_and_forecast_data(df_converter)
        final_df.iloc[:, 1:] = final_df.iloc[:, 1:].round(1)
        lang_table = processor.get_language_mapping()
        processor.update_final_sheet(final_df)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()