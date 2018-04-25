component AdverseDrugReactionApp {
  autoconnect port;
  port 
    in Image barcode,
    out String information;
  
  component MobileEHealthApp
    eHealthApp [bc2Service -> bcSer.Image];
  component BarcodeScannerService
    bcSer [String -> eHealthApp.bcAsString];
  component EHealtServer;
  component AdverseReactionDataBase;
  
  connect EHealtServer.result -> eHealthApp.answer;
}
