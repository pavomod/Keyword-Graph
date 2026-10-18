# Inconsistency-only analysis: ceps_mutated

| Strategy | Clusters | Findings | Distinct pairs | Input tok | Output tok | Thinking tok | Total tok | s |
|---|---|---|---|---|---|---|---|---|
| baseline | 12 | 20 | 28 | 12636 | 2082 | 11643 | 26361 | 138.4 |
| flat | 1 | 12 | 12 | 12359 | 1029 | 9945 | 23333 | 96.2 |

Pairs found by both: 7; only baseline: 21; only flat: 5.

## baseline

1. **59 – 121** [logical], cluster 0: Requirement 59 states that the Cancel Last Purchase transaction is performed off-line without issuer approval, whereas 121 mandates that it must always be performed online with direct card issuer approval.
2. **97 – 174** [logical], cluster 0: Requirement 97 states that POS devices are not required to support the cancel last purchase command, while 174 mandates that all POS devices must support it with no option to opt out.
3. **53 – 119** [logical], cluster 0: Requirement 53 requires the terminal to validate the card and sign the cancellation, but 119 states the transaction must be initiated without any involvement from the POS device.
4. **139 – 148** [constraint], cluster 1: Requirement 139 mandates that the card must maintain a global total balance with an absolute maximum limit, while 148 explicitly states there is no requirement for the card to maintain a maximum total value.
5. **102 – 134 – 166** [temporal], cluster 2: Requirement 102 states that S1 generation and transmission must occur after the transaction is completed. However, 134 and 166 state that S1 is sent in an authorization request and validated by the issuer to determine whether to allow the transaction to complete, which must happen before completion.
6. **103 – 123 – 124** [logical], cluster 2: Requirement 103 dictates that S1 must be generated entirely by the load device with no input from the card, whereas 123 and 124 state that the card generates S1 using data from both the card and the load device.
7. **160 – 184** [logical], cluster 2: Requirement 160 states that the issuer validates S1, but 184 strictly forbids the card issuer from performing S1 validation, mandating an independent third-party service instead.
8. **86 – 156** [logical], cluster 3: Requirement 86 permits certain POS devices to aggregate transactions, while 156 strictly forbids any automated aggregation processes in any POS device, requiring completely manual gathering.
9. **156 – 173** [logical], cluster 3: Requirement 173 allows the POS device to initiate the collection of a batch, which contradicts 156's mandate that the gathering of transaction records must be completely manual with no automated processes.
10. **153 – 182** [constraint], cluster 4: Requirement 153 states the count of aggregated transactions must be increased by 1, whereas 182 incorrectly mandates increasing the total count by the monetary amount of the aggregated transaction.
11. **150 – 151** [logical], cluster 5: Requirement 150 states a reversal is considered successful even with a negative response from the CEP card, but 151 states that under certain conditions the command is not allowed by the card, which contradicts the unconditional success described in 150.
12. **146 – 147** [logical], cluster 5: Requirement 146 states the CEP card authenticates the PSAM using previously received certificates and the S2 MAC, while 147 states the CEP card authenticates the terminal using the S2 MAC and the session key from the cancelled purchase transaction.
13. **82 – 94** [logical], cluster 6: Requirement 94 states that payment for a transaction is only required when a detail transaction is submitted, which contradicts 82, which allows acquirers to pay merchants based on header validation if transaction detail is missing.
14. **111 – 125** [constraint], cluster 8: Requirement 111 leaves the specification of keys and signatures for on-line authentication up to the issuer, whereas 125 strictly mandates that the issuer and card must share a secret key.
15. **141 – 142** [logical], cluster 9: Requirement 141 mandates a globally shared signature key identical across all cards, while 142 requires a unique diversified secret key personalized into each card.
16. **54 – 192 – 195** [constraint], cluster 9: Requirement 54 mandates the use of asymmetric cryptography for off-line transactions, but 192 and 195 require the terminal to generate and retain a symmetric MACing key signature.
17. **39 – 195** [logical], cluster 9: Requirement 39 states the card plays only a passive validation role while the merchant generates the primary signature, but 195 requires retaining the card's symmetric MACing key signature, implying the card actively generates a signature.
18. **14 – 23** [logical], cluster 10: Requirement 14 mandates that authorization of funds for load transactions must require a form of cardholder verification, whereas 23 states that cardholder verification is not necessary.
19. **117 – 212** [logical], cluster 10: Requirement 117 states that unload must not be precluded from being supported on any or all devices, but 212 restricts unloads to only be performed at a load device under the control of the card issuer.
20. **13 – 64 – 181** [logical], cluster 11: Requirement 13 mandates that aggregation records must be permanently stored on the card itself with no option for external storage, while 64 and 181 require aggregation records to be created and stored externally in the PSAM or POS device.

## flat

1. **38 – 39** [logical], cluster 0: Requirement 38 states that the CEP card must generate a signature during a purchase transaction, whereas 39 states the merchant generates the primary signature and the card plays only a passive validation role.
2. **11 – 119** [logical], cluster 0: Requirement 11 states that the POS device initiates the cancel last purchase transaction, while 119 states it can only be initiated by the cardholder without any involvement from the POS device.
3. **59 – 121** [logical], cluster 0: Requirement 59 states that the Cancel Last Purchase transaction is performed off-line without card issuer authentication, whereas 121 mandates that it must always be performed online with direct card issuer approval.
4. **103 – 123** [logical], cluster 0: Requirement 103 dictates that S1 must be generated entirely by the load device with no card participation, but 123 requires the card to generate S1 using both card and load device data.
5. **139 – 148** [logical], cluster 0: Requirement 139 requires the card to maintain a global total balance with a mandatory absolute maximum limit, while 148 explicitly states there is no requirement for the card to maintain a maximum total value.
6. **141 – 142** [logical], cluster 0: Requirement 141 mandates a globally shared signature key identical across all cards, whereas 142 requires a unique diversified secret key personalized into each card.
7. **113 – 154** [logical], cluster 0: Requirement 113 states the aggregate funds pool must be managed without centralized tracking by the card issuer, but 154 requires the card issuer to reflect all changes in the aggregate funds pool on their own systems.
8. **13 – 181** [logical], cluster 0: Requirement 13 restricts the storage of aggregation records exclusively to the card itself, while 181 requires them to be stored in the PSAM and allows storage in the POS device.
9. **97 – 174** [logical], cluster 0: Requirement 97 makes the cancel last purchase command optional for POS devices, whereas 174 mandates its support for all transactions with no option to opt out.
10. **134 – 184** [logical], cluster 0: Requirement 134 states that the card issuer validates S1, but 184 explicitly forbids the card issuer from doing so, requiring an independent third-party authentication service instead.
11. **102 – 166** [temporal], cluster 0: Requirement 102 states that S1 generation and transmission must occur after the transaction is completed, while 166 states S1 is sent in an authorization request, which must happen before the transaction completes.
12. **156 – 182** [logical], cluster 0: Requirement 156 forbids automated aggregation processes in any POS device or system, whereas 182 requires the PSAM (part of the POS system) to automatically increase the total count and amount for aggregated transactions.

## Requirements cited

- **11**: After the CEP card is inserted in the POS device, the POS device initiates the cancel last purchase transaction
- **13**: Aggregation records must be permanently stored on the card itself, with no option to store them in external systems or devices.
- **14**: All load transactions are on-line transactions. Authorization of funds for load transactions must require a form of cardholder verification. The load device must support on-line encrypted PIN or off-line PIN verification
- **23**: Cardholder verification is not necessary, but may be done
- **38**: During a purchase transaction, the CEP card must ensure that it is dealing with an authentic POS device, and it must generate a signature to allow the card issuer to verify the integrity of the transaction
- **39**: During a purchase transaction, the merchant must generate the primary transaction signature, with the card playing only a passive validation role.
- **53**: For cancel last purchase transactions, the terminal must validate the card and sign the cancellation
- **54**: For increased security and convenience, asymmetric cryptography must be used as the authentication security for off-line transactions
- **59**: However, unlike the load, the Cancel Last Purchase is performed off-line without opportunity for the card issuer to authenticate the card or to approve the transaction
- **64**: If the business rules allow this transaction is to be aggregated, the PSAM must determine if it has an aggregation record for the card issuer. If the PSAM does not have an aggregation record for the card issuer, a new aggregation record for the card issuer must be created
- **82**: Merchant acquirers may institute pay on header arrangements for providing payment to their merchants. In this arrangement, the merchant acquirer relies on the header validation to determine payment to the merchant if transaction detail is missing due to some problem. The specific rules covering pay on header arrangements are outside the scope of this document
- **86**: Only certain POS devices, which have been installed at merchants approved for aggregation by both the scheme provider and merchant acquirer, must be permitted to aggregate transactions
- **94**: Payment for a transaction is only required when a detail transaction is submitted for payment, A merchant acquirer may choose to pay the merchant using the batch total
- **97**: POS devices are not required to support the cancel last purchase command. If this command is supported, the POS device must have security to prevent unauthorized or fraudulent use of the transaction
- **102**: S1 generation and transmission must occur after the transaction is completed, serving as a post-transaction verification mechanism.
- **103**: S1 must be generated entirely by the load device, with no input or participation from the card itself, and sent directly to the card issuer.
- **111**: Specification of keys and signatures used for CEPS on-line authentication is left to the issuer, however MACs passed over shared networks must be no more than 8 bytes long, binary format
- **113**: The aggregate funds pool must be managed dynamically by individual merchants, without centralized tracking by the card issuer.
- **117**: The architecture for the electronic purse should not preclude the eventual deployment of a variety of devices to meet the needs of many card acceptance environments. Unload must not be precluded from being supported on any or all of these devices at the option of the acquirer
- **119**: The cancel last purchase transaction can only be initiated by the cardholder, without any involvement from the POS device.
- **121**: The cancel last purchase transaction must always be performed online, requiring direct card issuer approval and authentication.
- **123**: The card generates S1 using card and load device data, and sends S1 to the load device to be forwarded to the issuer
- **124**: The card generates S1, using data from the card and load device, and sends S1 to the card issuer to authenticate the card and the data that is to be used in the transaction
- **125**: The card issuer and the CEP card share a secret key to generate and verify MACs
- **134**: The card issuer validates S1, calculates the amount of the transaction into the new currency, and determines whether to allow the exchange to complete
- **139**: The card must maintain a global total balance across all currency slots, with a mandatory absolute maximum limit that cannot be exceeded regardless of individual slot policies.
- **141**: The card must use a globally shared signature key that is identical across all cards, eliminating the concept of personalized authentication.
- **142**: The card should use a unique diversified secret key, personalized into the card, to generate and authenticate transaction signatures
- **146**: The CEP card authenticates the PSAM using the certificates previously received and the S2, The CEP card increments the value of the purse and logs the transaction
- **147**: The CEP card authenticates the terminal using the S2 MAC and the session key from the purchase transaction being cancelled. The CEP card increments the value of the purse and logs the transaction. The CEP card verifies that the cancellation amount is equal to the amount of the last step of the purchase transaction
- **148**: The CEP card limits each slot to a maximum balance. The maximum balance for a slot is established when a currency is assigned to the slot, and is determined externally by the card issuer or by CEP card data which indicates a maximum balance in a reference currency. There is no requirement that all slot maximums have the same relative value, and there is no requirement for the card to maintain a maximum total value
- **150**: The CEP card sends a response to the reversal command to the POS device. The reversal is considered to be successful even in the event of a negative response or no response from the CEP card
- **151**: The CEP card verifies that the PSAM is the PSAM used in the original transaction, computes a S1 MAC using the same DES session key that was used for the purchase transaction being cancelled, and responds to the POS device with the S1 and the identification of the last transaction. If the purchase transaction to be canceled has been canceled or is not part of the active batch, the command is not allowed
- **153**: The count of aggregated transactions in the aggregation record must be increased by 1
- **154**: The electronic purse card issuer must reflect all changes in the value of a loaded card and the aggregate funds pool on their card databases and funds pool accounting systems (where separate)
- **156**: The gathering of transaction records must be completely manual, with no automated aggregation processes allowed in any POS device or system.
- **160**: The issuer validates S1 to authenticate the card and the data to be used in the transaction
- **166**: The load device sends S1 in an authorization request to the issuer
- **173**: The POS device closes the batch when the batch is collected. Collection may be initiated by the POS device, the merchant, or by the merchant acquirer
- **174**: The POS device is mandated to support the cancel last purchase command for all transactions, with no option to opt out.
- **181**: The PSAM must create a MAC on the updated aggregation record. Aggregation records by card issuer must be stored in the PSAM. They may also be stored in the POS device as well
- **182**: The PSAM must increase the total count and amount of the active batch by the amount of the aggregated transaction
- **184**: The S1 validation process must be performed by an independent third-party authentication service, not by the card issuer.
- **192**: The terminal must generate a symmetric MACing key signature and send it to the merchant acquirer
- **195**: The terminal must secure transactions by retaining the card’s symmetric MACing key signature as part of the transaction
- **212**: Unloads are permitted only to an account controlled by the card issuer and are performed only at a load device under the control of the card issuer
