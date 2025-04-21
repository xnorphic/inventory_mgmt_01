import pandas as pd
import numpy as np

class InventoryOptimizer:
    def __init__(self):
        self.data = None

    def load_dataframe(self, df):
        self.data = df.copy()
        return self

    def calculate_monthly_avg_sales(self):
        self.data['monthly_avg_3m'] = self.data['last_3month_sold'] / 3
        self.data['monthly_avg_6m'] = self.data['last_6month_sold'] / 6
        self.data['monthly_avg_weighted'] = (self.data['monthly_avg_3m'] * 0.7 + self.data['monthly_avg_6m'] * 0.3)
        return self

    def forecast_sales(self, forecast_periods=3):
        self.data['forecast_avg'] = self.data['monthly_avg_weighted']
        for i in range(1, forecast_periods + 1):
            self.data[f'forecast_{i}'] = self.data['forecast_avg']
        return self

    def calculate_stock_needed(self, revenue_target):
        self.data['forecast_revenue'] = self.data['forecast_avg'] * self.data['selling_price_per_unit']
        total_forecast_revenue = self.data['forecast_revenue'].sum()
        scaling_factor = revenue_target / total_forecast_revenue if total_forecast_revenue > 0 else 1
        scaled_sales = self.data['forecast_avg'] * scaling_factor
        self.data['stock_needed_for_target'] = (scaled_sales * 1.3).round().astype(int)
        self.data['additional_stock_needed'] = np.maximum(0, self.data['stock_needed_for_target'] - self.data['current_stock'])
        self.data['procurement_cost'] = self.data['additional_stock_needed'] * self.data['cogs_per_unit']
        return self

    def calculate_reorder_points(self, service_level=0.95):
        self.data['daily_sales_rate'] = self.data['forecast_avg'] / 30
        z_score = {0.9: 1.28, 0.95: 1.65, 0.98: 2.05, 0.99: 2.33}.get(service_level, 1.65)
        self.data['daily_demand_std'] = self.data['daily_sales_rate'] * 0.3
        self.data['safety_stock'] = z_score * self.data['daily_demand_std'] * np.sqrt(self.data['lead_time_days'])
        self.data['reorder_point'] = (self.data['daily_sales_rate'] * self.data['lead_time_days'] + self.data['safety_stock']).round().astype(int)
        self.data['days_until_stockout'] = np.where(
            self.data['daily_sales_rate'] > 0,
            self.data['current_stock'] / self.data['daily_sales_rate'],
            999
        )
        self.data['needs_reorder'] = self.data['current_stock'] <= self.data['reorder_point']
        self.data['ideal_order_qty'] = np.maximum(0, (
            self.data['reorder_point'] + self.data['daily_sales_rate'] * 30 - self.data['current_stock']
        ).round().astype(int))
        self.data['recommended_order_qty'] = np.where(
            self.data['ideal_order_qty'] > 0,
            np.maximum(
                self.data['moq'],
                np.ceil(self.data['ideal_order_qty'] / self.data['moq']) * self.data['moq']
            ),
            0
        ).astype(int)
        return self

    def generate_report(self, revenue_target):
        critical_items = self.data[self.data['days_until_stockout'] < 14].copy()
        high_value_items = self.data.sort_values('procurement_cost', ascending=False).head(10)
        metrics = {
            'total_skus': len(self.data),
            'total_current_stock_value': (self.data['current_stock'] * self.data['cogs_per_unit']).sum(),
            'monthly_revenue_potential': self.data['forecast_revenue'].sum(),
            'skus_needing_reorder': self.data['needs_reorder'].sum(),
            'out_of_stock_skus': (self.data['current_stock'] == 0).sum(),
            'low_stock_skus': ((self.data['current_stock'] > 0) & (self.data['current_stock'] <= self.data['monthly_avg_weighted'] * 0.5)).sum(),
        }
        return {
            'metrics': metrics,
            'stock_recommendations': self.data.sort_values('additional_stock_needed', ascending=False),
            'reorder_recommendations': self.data[self.data['needs_reorder']].sort_values('days_until_stockout'),
            'critical_items': critical_items,
            'high_value_items': high_value_items
        }
