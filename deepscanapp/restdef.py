from __future__ import annotations
from typing import Optional, Set
from typing import List, Optional
from pydantic import BaseModel
# 1.1 define REST-structure with objects
# response for Prediction


#response Model for
class Decision(BaseModel):
    transactionid: str
    prediction:  float
    probability: float
    confidence:  float
    prediction2:  float
    probability2: float
    confidence2:  float

# response for Test


class Trainresult(BaseModel):
    success: str


###########################################################
#### Data Input / Observation with nested submodels
# can be generated via tools from a json


class DeliveryAddress(BaseModel):
    company: str
    countryCode: str


class InvoiceAddress(BaseModel):
    company: str
    countryCode: str


class PurchaseOrderItem(BaseModel):
    id: str
    version: str
    addPageBreakBefore: bool
    articleId: str
    articleNumber: str
    createdDate: int
    customAttributes: List
    description: Optional[str] = None
    discountPercentage: str
    freeTextItem: bool
    grossAmount: str
    grossAmountInCompanyCurrency: str
    lastModifiedDate: int
    manualUnitPrice: bool
    netAmount: str
    netAmountForStatistics: str
    netAmountForStatisticsInCompanyCurrency: str
    netAmountInCompanyCurrency: str
    positionNumber: int
    quantity: str
    receivedQuantity: str
    reductionAdditionItems: List
    supplierArticleId: Optional[str] = None
    taxId: str
    taxName: str
    title: str
    unitId: str
    unitName: str
    unitPrice: str
    unitPriceInCompanyCurrency: str


class RecordAddress(BaseModel):
    city: str
    company: str
    countryCode: str
    street1: str
    zipcode: str


class StatusHistoryItem(BaseModel):
    status: str
    statusDate: int
    userId: str


class ResultItem(BaseModel):
    id: str
    version: str
    createdDate: int
    customAttributes: List
    deliveryAddress: DeliveryAddress
    grossAmount: str
    grossAmountInCompanyCurrency: str
    headerDiscount: str
    headerSurcharge: str
    invoiceAddress: InvoiceAddress
    lastModifiedDate: int
    netAmount: str
    netAmountInCompanyCurrency: str
    orderDate: int
    orderDescription: str
    paymentMethodId: Optional[str] = None
    paymentMethodName: Optional[str] = None
    plannedDeliveryDate: int
    purchaseOrderItems: List[PurchaseOrderItem]
    purchaseOrderNumber: str
    purchaseOrderType: str
    received: bool
    recordAddress: RecordAddress
    recordCurrencyId: str
    recordCurrencyName: str
    responsibleUserId: str
    responsibleUserUsername: str
    sentToRecipient: bool
    servicePeriodFrom: Optional[int] = None
    servicePeriodTo: Optional[int] = None
    shipmentMethodId: Optional[str] = None
    shipmentMethodName: Optional[str] = None
    shippingCostItems: List
    status: str
    statusHistory: List[StatusHistoryItem]
    supplierId: str
    supplierNumber: str
    tags: List
    termOfPaymentId: str
    termOfPaymentName: str
    warehouseId: str
    warehouseName: str


class PurReq(BaseModel):
    result: Optional[List[ResultItem]] = None
